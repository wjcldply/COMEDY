# Dataset Preparation
- Loading `Nuo97/Dolphin-Task1` & `Nuo97/Dolphin-Task2` & `Nuo97/Dolphin-Task3`
- Mixing (MultiTask)
    - Mix 3 Datasets on Task1~3 to Create a Single, Concatenated Dataset for Multi-Task Training
- Shuffling Dataset (MultiTask-Shuffle)
- Splitting `Train` & `Validation`

```bash
$ python build_dataset.py
```