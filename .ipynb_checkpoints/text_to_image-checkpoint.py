import os
import glob
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

from watermarker import SD3Text2ImgWatermarker
from utils import apply_attack


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
# COCO Prompt Loader
# ============================================================

def load_coco_prompts(image_dir, annotation_path):
    image_paths = glob.glob(os.path.join(image_dir, "*"))

    with open(annotation_path, "r", encoding="utf-8") as f:
        annotations_data = json.load(f)

    # 建立 image_id -> caption 映射（避免 O(N²) 搜索）
    id_to_caption = {}
    for anno in annotations_data["annotations"]:
        image_id = anno["image_id"]
        if image_id not in id_to_caption:
            id_to_caption[image_id] = anno["caption"]

    labels, prompts = [], []

    for path in image_paths:
        image_id = int(os.path.basename(path).split(".")[0])
        if image_id in id_to_caption:
            labels.append(image_id)
            prompts.append(id_to_caption[image_id])

    return labels, prompts


# ============================================================
# MAIN BATCH PROCESS
# ============================================================

def batch_process(
    output_dir,
    model_path,
    img_num,
    message="FLOW",
    dataset="coco"
):

    os.makedirs(output_dir, exist_ok=True)

    marker = SD3Text2ImgWatermarker(
        model_path,
        strength=0.005,
        num_chars=len(message)
    )

    target_bits = marker._msg_to_bits(message)

    stats = {
        "score_clean": [],
        "score_wm": [],
        "score_attacks": {},
        "acc_none": [],
    }

    # --------------------------------------------------------
    # Helper Functions
    # --------------------------------------------------------

    def get_score(img):
        return marker.detect(img)["confidence_score"]

    def bit_accuracy(bits_pred):
        return sum(
            b1 == b2 for b1, b2 in zip(target_bits, bits_pred)
        ) / len(target_bits)

    def record_attack(name, img):
        stats["score_attacks"].setdefault(name, []).append(
            get_score(img)
        )

        _, bits = marker.extract(img)
        stats.setdefault(f"acc_{name}", []).append(
            bit_accuracy(bits)
        )

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    if dataset == "coco":
        image_dir = "/coco/dataset/path"
        annotation_path = "/coco/dataset/annotations/captions_val2017.json"
    
        labels, prompts = load_coco_prompts(
            image_dir,
            annotation_path
        )
    else:
        raise ValueError("Unsupported dataset")

    total_samples = min(img_num, len(prompts))
    print(f"[*] Generating {total_samples} samples...")

    # --------------------------------------------------------
    # Generation Loop
    # --------------------------------------------------------

    for i in tqdm(range(total_samples)):

        prompt = prompts[i]
        label = labels[i]

        # 1️⃣ Generate clean image
        clean_img = marker.original_generate(prompt, message)
        clean_img.save(
            os.path.join(output_dir, f"{label}_orig.png")
        )

        stats["score_clean"].append(get_score(clean_img))

        # 2️⃣ Generate watermarked image
        wm_img = marker.embed(
            prompt,
            message,
            denoising_strength=0.06
        )

        wm_img.save(
            os.path.join(output_dir, f"{label}_wm.png")
        )

        stats["score_wm"].append(get_score(wm_img))

        # 3️⃣ No-attack accuracy
        _, bits = marker.extract(wm_img)
        stats["acc_none"].append(bit_accuracy(bits))

        # 4️⃣ Attacks
        attacks = [
            ("jpeg_30", "jpeg", 30),
        ]

        for name, atk_type, param in attacks:
            attacked_img = apply_attack(wm_img, atk_type, param)
            record_attack(name, attacked_img)

    # ========================================================
    # REPORT
    # ========================================================

    print("\n" + "=" * 60)
    print(f"Batch Statistics (Samples: {total_samples})")
    print("=" * 60)

    # ---------------- Detection ----------------

    print("\n================ Detection =================")
    compute_metrics(
        stats["score_wm"],
        stats["score_clean"],
        "Original"
    )

    # ---------------- Robustness ----------------

    print("\n================ Robustness AUC ================")
    robust_auc = []

    for atk, scores in stats["score_attacks"].items():
        auc_val = compute_metrics(
            scores,
            stats["score_clean"],
            atk
        )
        robust_auc.append(auc_val)

    print("\nAverage Robustness AUC:",
          np.mean(robust_auc) if robust_auc else 0)

    # ---------------- Bit Accuracy ----------------

    print("\n📊 No Attack")
    print(f"Accuracy: {np.mean(stats['acc_none']) * 100:.2f}%")

    for key in stats:
        if key.startswith("acc_") and key != "acc_none":
            print(f"{key.replace('acc_', ''):<25}: "
                  f"{np.mean(stats[key]) * 100:.2f}%")

    print("\n" + "=" * 60)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    OUTPUT_FOLDER = "./output_generate"
    IMAGE_NUM = 1000
    DATASET = "coco"
    MODEL_WEIGHTS = "stabilityai/stable-diffusion-3-medium-diffusers"

    batch_process(
        OUTPUT_FOLDER,
        MODEL_WEIGHTS,
        IMAGE_NUM,
        message="SDFLOW",
        dataset=DATASET
    )