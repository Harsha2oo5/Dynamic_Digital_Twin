# Dataset audit report

- Dataset root: `C:\Users\harsh\Desktop\dynamic_digital_twin\digital twin project datasets`
- Audited: 2026-11-01T01:05:51
- Total images: **543** (readable: 543)
- Formats: .png ×540, .jpg ×3
- macOS metadata files ignored: 211
- Annotation files found: 0; labelled images: 0
- **Ready for detection training: NO**

## Images per class

| ID | Class | Folder | Images | Labelled | Formats | Width range | Height range | Most common size | Independent groups |
|---|---|---|---|---|---|---|---|---|---|
| 0 | antenna | antenna | 100 | 0 | .png ×100 | 198–1251 | 159–841 | 447x430 (×2) | 100 |
| 1 | router | Archive (7) | 74 | 0 | .png ×74 | 1206–1206 | 2622–2622 | 1206x2622 (×74) | 56 |
| 2 | cable | Archive (6) | 44 | 0 | .png ×42, .jpg ×2 | 300–1206 | 771–2622 | 1206x2622 (×42) | 41 |
| 3 | battery | Archive (5) | 62 | 0 | .png ×62 | 1206–1206 | 2622–2622 | 1206x2622 (×62) | 55 |
| 4 | power_supply_unit | Archive (4) | 93 | 0 | .png ×92, .jpg ×1 | 643–1206 | 673–2622 | 1206x2622 (×92) | 85 |
| 5 | equipment_cabinet | Equipment Cabinet | 70 | 0 | .png ×70 | 150–628 | 241–828 | 628x828 (×1) | 70 |
| 6 | radio_unit | radio unit | 100 | 0 | .png ×100 | 190–766 | 226–600 | 589x583 (×1) | 95 |

## Integrity

- Unreadable / corrupt images: 0
- Exact duplicate groups (identical bytes): 0
- Near-duplicate groups (dHash distance ≤ 30): 39 covering 84 images
- Near-duplicate groups spanning different classes: 4
- Unmapped folders: none

Near-duplicate groups are kept inside a single split by `prepare_dataset` to avoid leakage.
Cross-class groups are the same photo saved in two class folders; such scenes usually contain
several component types, so every instance must be boxed during annotation.

Cross-class groups:

- `Archive (4)/IMG_6444.PNG`, `Archive (4)/IMG_6473.PNG`, `Archive (4)/IMG_6516.PNG`, `Archive (5)/IMG_6328.PNG`
- `Archive (4)/IMG_6464.PNG`, `Archive (5)/IMG_6303.PNG`
- `Archive (4)/IMG_6475.PNG`, `Archive (4)/IMG_6517.PNG`, `Archive (5)/IMG_6310.PNG`
- `Archive (4)/IMG_6519.PNG`, `Archive (5)/IMG_6353.PNG`

## Blockers

- No bounding-box annotations exist. Folder names give an image-level class only; YOLO detection needs a box + class id for every object instance.
