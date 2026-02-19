import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from fer import FER


# =========================
# Utils
# =========================
def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def rotate_image(img, angle: int):
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("angle must be one of 0, 90, 180, 270")


def map_box_back(x, y, w, h, angle, orig_w, orig_h):
    if angle == 0:
        return x, y, w, h
    if angle == 180:
        return orig_w - (x + w), orig_h - (y + h), w, h
    if angle == 90:
        xo = orig_w - (y + h)
        yo = x
        return xo, yo, h, w
    if angle == 270:
        xo = y
        yo = orig_h - (x + w)
        return xo, yo, h, w
    raise ValueError("angle must be one of 0, 90, 180, 270")


def nms_boxes(boxes, overlap_thresh=0.30):
    if not boxes:
        return []
    rects = []
    for (x, y, w, h) in boxes:
        rects.append([x, y, x + w, y + h])

    rects = np.array(rects, dtype=np.float32)
    x1 = rects[:, 0]
    y1 = rects[:, 1]
    x2 = rects[:, 2]
    y2 = rects[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    pick = []
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / areas[idxs[:-1]]
        idxs = idxs[np.where(overlap <= overlap_thresh)[0]]

    out = []
    for i in pick:
        x, y, xx2, yy2 = rects[i]
        out.append((int(x), int(y), int(xx2 - x), int(yy2 - y)))
    return out


def create_yunet(model_path: str, input_size: tuple[int, int], score_threshold: float):
    w, h = input_size
    return cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(w, h),
        score_threshold=score_threshold,
        nms_threshold=0.30,
        top_k=5000,
    )


def yunet_detect(detector, frame_bgr):
    h, w = frame_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame_bgr)

    dets = []
    if faces is None:
        return dets

    for f in faces:
        x, y, bw, bh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
        score = float(f[-1])
        if bw > 0 and bh > 0:
            dets.append((x, y, bw, bh, score))
    return dets


def detect_faces_yunet_multirotation(frame_bgr, detector, min_size=(50, 50), ratio_min=0.70, ratio_max=1.45):
    orig_h, orig_w = frame_bgr.shape[:2]
    best = []

    for angle in (0, 90, 180, 270):
        rotated = rotate_image(frame_bgr, angle)
        dets = yunet_detect(detector, rotated)

        mapped = []
        for (x, y, w, h, score) in dets:
            if w < min_size[0] or h < min_size[1]:
                continue

            ratio = w / float(h)
            if not (ratio_min <= ratio <= ratio_max):
                continue

            xo, yo, wo, ho = map_box_back(x, y, w, h, angle, orig_w, orig_h)

            xo = max(0, min(xo, orig_w - 1))
            yo = max(0, min(yo, orig_h - 1))
            wo = max(1, min(wo, orig_w - xo))
            ho = max(1, min(ho, orig_h - yo))

            mapped.append((xo, yo, wo, ho, score))

        if len(mapped) > len(best):
            best = mapped

    return best


# =========================
# Movement / Anomaly + Activity (simple)
# =========================
def compute_motion_score(prev_gray, curr_gray):
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff))


def classify_activity(motion_score, mean, std, z_anomaly=3.0):
    if std <= 1e-6:
        z = 0.0
    else:
        z = (motion_score - mean) / std

    if z > z_anomaly:
        return "movimento_brusco", True, z
    # regra simples: acima da média é movimento, abaixo é parado
    if motion_score > mean:
        return "movimento", False, z
    return "parado", False, z


# =========================
# Main
# =========================
def main(
    input_path: str = "assets/Unlocking_Facial_Recognition_ Diverse_Activities_Analysis.mp4",
    model_path: str = "assets/face_detection_yunet_2023mar.onnx",
    output_video_path: str = "outputs/annotated.mp4",
    report_txt_path: str = "outputs/report.txt",
    report_json_path: str = "outputs/report.json",
    events_csv_path: str = "outputs/events.csv",
    # detector tuning
    yunet_score_threshold: float = 0.75,
    resize_scale: float = 0.75,
    min_face_size_small: tuple[int, int] = (45, 45),
    nms_overlap: float = 0.30,
    # emotion tuning
    emotion_every_n_frames: int = 10,
    # anomaly tuning
    motion_window: int = 90,         # janela para média/desvio (frames)
    anomaly_z_threshold: float = 3.0 # z-score
) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Vídeo não encontrado: {input_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo YuNet não encontrado: {model_path}\n"
            "Baixe face_detection_yunet_2023mar.onnx e coloque em assets/."
        )

    ensure_dir(os.path.dirname(output_video_path))
    ensure_dir(os.path.dirname(report_txt_path))
    ensure_dir(os.path.dirname(report_json_path))
    ensure_dir(os.path.dirname(events_csv_path))

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    detector = create_yunet(model_path, (width, height), score_threshold=yunet_score_threshold)

    # Emoção (FER)
    # mtcnn=False para ser mais leve/estável; se quiser mais recall, troque para True (mais lento)
    emotion_model = FER(mtcnn=True)

    # métricas
    frames_analyzed = 0
    anomalies_count = 0

    emotion_counts = {}
    activity_counts = {"parado": 0, "movimento": 0, "movimento_brusco": 0}

    # movimento
    prev_gray_full = None
    motion_scores = []

    # para suavizar emoção por face "daquele frame"
    last_emotion_for_face_idx = {}

    # eventos para CSV
    events = []

    pbar = tqdm(total=total_frames_video if total_frames_video > 0 else None, desc="Processando")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frames_analyzed += 1

        # =========================
        # Resize para detectar faces pequenas
        # =========================
        if resize_scale and resize_scale != 1.0:
            small = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)
        else:
            small = frame

        # Faces (YuNet + multi-rotação) no SMALL
        dets = detect_faces_yunet_multirotation(
            small,
            detector,
            min_size=min_face_size_small
        )

        # NMS (small coords)
        boxes_small = [(x, y, w, h) for (x, y, w, h, s) in dets]
        boxes_small = nms_boxes(boxes_small, overlap_thresh=nms_overlap)

        inv = (1.0 / resize_scale) if (resize_scale and resize_scale != 1.0) else 1.0

        # Converte para coords do frame original
        final_faces = []
        for (x, y, w, h) in boxes_small:
            xo = int(x * inv)
            yo = int(y * inv)
            wo = int(w * inv)
            ho = int(h * inv)

            xo = max(0, min(xo, width - 1))
            yo = max(0, min(yo, height - 1))
            wo = max(1, min(wo, width - xo))
            ho = max(1, min(ho, height - yo))

            # score aproximado
            best_s = 0.0
            for (xx, yy, ww, hh, s) in dets:
                if abs(xx - x) + abs(yy - y) + abs(ww - w) + abs(hh - h) < 50:
                    best_s = max(best_s, float(s))

            final_faces.append((xo, yo, wo, ho, best_s))

        # =========================
        # Emoções (a cada N frames)
        # =========================
        emotions_in_frame = []
        do_emotion = (frames_analyzed % emotion_every_n_frames == 0)

        for idx, (x, y, w, h, s) in enumerate(final_faces, start=1):
            emotion = last_emotion_for_face_idx.get(idx, "unknown")

            if do_emotion:
                face_roi = frame[y:y + h, x:x + w]
                # evita ROI vazia
                if face_roi.size > 0:
                    try:
                        r = emotion_model.detect_emotions(face_roi)
                        if r:
                            emotions = r[0]["emotions"]
                            emotion = max(emotions, key=emotions.get)
                    except Exception:
                        emotion = emotion  # mantém anterior

                last_emotion_for_face_idx[idx] = emotion

            emotions_in_frame.append(emotion)
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + (1 if do_emotion else 0)

            # desenha bbox + emoção
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Face {idx} | {emotion}",
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # =========================
        # Movimento + anomalia + atividade
        # =========================
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        activity = "parado"
        is_anomaly = False
        z = 0.0
        motion_score = 0.0

        if prev_gray_full is not None:
            motion_score = compute_motion_score(prev_gray_full, gray_full)
            motion_scores.append(motion_score)
            if len(motion_scores) > motion_window:
                motion_scores = motion_scores[-motion_window:]

            mean = float(np.mean(motion_scores)) if motion_scores else 0.0
            std = float(np.std(motion_scores)) if motion_scores else 0.0

            activity, is_anomaly, z = classify_activity(
                motion_score, mean, std, z_anomaly=anomaly_z_threshold
            )

            activity_counts[activity] += 1

            if is_anomaly:
                anomalies_count += 1
                cv2.putText(
                    frame,
                    "ANOMALIA!",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
        else:
            # primeiro frame
            activity_counts["parado"] += 1

        prev_gray_full = gray_full

        # overlays
        cv2.putText(
            frame,
            f"Frame: {frames_analyzed} | Faces: {len(final_faces)} | Activity: {activity}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # salva evento (CSV)
        events.append({
            "frame": frames_analyzed,
            "time_s": round(frames_analyzed / fps, 3),
            "faces": len(final_faces),
            "emotions": "|".join(emotions_in_frame) if emotions_in_frame else "",
            "activity": activity,
            "motion_score": round(motion_score, 6),
            "motion_z": round(float(z), 4),
            "anomaly": int(is_anomaly),
        })

        out.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    # =========================
    # Resumo
    # =========================
    dominant_emotion = "unknown"
    if emotion_counts:
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)

    dominant_activity = max(activity_counts, key=activity_counts.get) if activity_counts else "unknown"

    duration_s = frames_analyzed / fps if fps else 0.0

    report = {
        "total_frames_analyzed": int(frames_analyzed),
        "duration_s": float(duration_s),
        "fps": float(fps),
        "anomalias_detectadas": int(anomalies_count),
        "emocao_predominante": dominant_emotion,
        "atividade_predominante": dominant_activity,
        "contagem_emocoes": emotion_counts,
        "contagem_atividades": activity_counts,
        "outputs": {
            "annotated_video": output_video_path,
            "events_csv": events_csv_path,
            "report_txt": report_txt_path,
            "report_json": report_json_path,
        }
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # relatório TXT (para colar no PDF final)
    lines = []
    lines.append("RESUMO AUTOMÁTICO - TECH CHALLENGE\n")
    lines.append(f"Total de frames analisados: {frames_analyzed}")
    lines.append(f"Número de anomalias detectadas: {anomalies_count}")
    lines.append(f"Emoção predominante: {dominant_emotion}")
    lines.append(f"Atividade predominante: {dominant_activity}")
    lines.append("")
    lines.append("Contagem de atividades:")
    for k, v in sorted(activity_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Contagem de emoções (amostradas):")
    for k, v in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v}")

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # CSV dos eventos
    df = pd.DataFrame(events)
    df.to_csv(events_csv_path, index=False, encoding="utf-8")

    print("\nOK!")
    print(f"Vídeo anotado: {output_video_path}")
    print(f"Report TXT:    {report_txt_path}")
    print(f"Report JSON:   {report_json_path}")
    print(f"Events CSV:    {events_csv_path}")


if __name__ == "__main__":
    main()
