import os
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

from watermarker import SD3FlowTrajectoryWatermarker
from utils import calculate_psnr, apply_attack


# ============================================================
# ROC METRICS
# ============================================================

def compute_metrics(pos_scores, neg_scores, name=""):
    labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    scores = np.array(pos_scores + neg_scores)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    target_fpr = 0.01
    tpr_at_1 = np.interp(target_fpr, fpr, tpr)

    print(f"\n===== {name} =====")
    print(f"AUC: {roc_auc:.4f}")
    print(f"TPR @ 1% FPR: {tpr_at_1:.4f}")

    return roc_auc


# ============================================================
# MAIN BATCH PROCESS
# ============================================================

def batch_process(input_dir, output_dir, model_path, message="FLOW"):

    os.makedirs(output_dir, exist_ok=True)

    # 初始化水印器
    marker = SD3FlowTrajectoryWatermarker(
        model_path,
        strength=0.06,
        num_chars=len(message)
    )

    target_bits = marker._msg_to_bits(message)

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ][:2]

    print(f"[*] Found {len(image_files)} images. Start processing...")

    # ========================================================
    # Statistics Container
    # ========================================================

    stats = {
        "psnr": [],
        "score_clean": [],
        "score_wm": [],
        "score_attacks": {},
        "acc_none": [],
        "acc_jpeg_30": [],
    }

    # --------------------------------------------------------

    def get_detection(img):
        return marker.detect(img)

    def get_score(img):
        return get_detection(img)["confidence_score"]

    def bit_accuracy(bits_pred):
        return sum(
            b1 == b2 for b1, b2 in zip(target_bits, bits_pred)
        ) / len(target_bits)

    def record_attack_score(name, img):
        stats["score_attacks"].setdefault(name, []).append(
            get_score(img)
        )

    # ========================================================
    # Processing Loop
    # ========================================================

    for filename in tqdm(image_files):

        img_path = os.path.join(input_dir, filename)
        save_path = os.path.join(
            output_dir,
            os.path.splitext(filename)[0] + ".png"
        )

        # ----------------------------------------------------
        # 1. Embed Watermark
        # ----------------------------------------------------

        wm_img = marker.embed(
            img_path,
            message,
            denoising_strength=0.06
        )

        orig_img = Image.open(img_path).convert("RGB")
        orig_size = orig_img.size

        wm_img = wm_img.resize(orig_size, Image.Resampling.LANCZOS)
        wm_img.save(save_path)

        # ----------------------------------------------------
        # 2. Imperceptibility
        # ----------------------------------------------------

        psnr_val = calculate_psnr(orig_img, wm_img, img_size=256)
        stats["psnr"].append(psnr_val)

        # ----------------------------------------------------
        # 3. Detection Scores
        # ----------------------------------------------------

        stats["score_clean"].append(get_score(orig_img))
        stats["score_wm"].append(get_score(wm_img))

        # ----------------------------------------------------
        # 4. Bit Accuracy (No Attack)
        # ----------------------------------------------------

        _, bits_none = marker.extract(wm_img)
        stats["acc_none"].append(bit_accuracy(bits_none))

        # ----------------------------------------------------
        # 5. Attacks
        # ----------------------------------------------------

        attacks = [
            ("jpeg_30", "jpeg", 30),
        ]

        for name, atk_type, param in attacks:

            attacked_img = apply_attack(wm_img, atk_type, param)

            # robustness AUC score
            record_attack_score(name, attacked_img)

            # bit accuracy
            _, bits_attacked = marker.extract(attacked_img)
            stats[f"acc_{name}"] = stats.get(f"acc_{name}", [])
            stats[f"acc_{name}"].append(bit_accuracy(bits_attacked))

    # ========================================================
    # Report
    # ========================================================

    print("\n" + "=" * 60)
    print(f"Batch Statistics (Samples: {len(image_files)})")
    print("=" * 60)

    # --------------------------------------------------------
    # Imperceptibility
    # --------------------------------------------------------

    print("\n📊 Imperceptibility")
    print(f"Average PSNR: {np.mean(stats['psnr']):.2f} dB")

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    print("\n================ Detection =================")
    compute_metrics(
        stats["score_wm"],
        stats["score_clean"],
        "Original"
    )

    # --------------------------------------------------------
    # Robustness
    # --------------------------------------------------------

    print("\n================ Robustness AUC ================")

    robust_auc = []
    for atk, scores in stats["score_attacks"].items():
        auc_val = compute_metrics(scores, stats["score_clean"], atk)
        robust_auc.append(auc_val)

    print("\nAverage Robustness AUC:",
          np.mean(robust_auc) if robust_auc else 0)

    # --------------------------------------------------------
    # Bit Accuracy
    # --------------------------------------------------------

    print("\n📊 No Attack")
    print(f"Accuracy: {np.mean(stats['acc_none']) * 100:.2f}%")

    print("\n📊 JPEG Compression")
    print(f"JPEG-30: {np.mean(stats['acc_jpeg_30']) * 100:.2f}%")

    print("\n" + "=" * 60)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    INPUT_FOLDER = "/Your/original/image/folder"
    OUTPUT_FOLDER = "./output"
    MODEL_WEIGHTS = "stabilityai/stable-diffusion-3-medium-diffusers"
    
    batch_process(
        INPUT_FOLDER,
        OUTPUT_FOLDER,
        MODEL_WEIGHTS,
        message="SDFLOW"
    )