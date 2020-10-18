# Smoker-Classifier

1. 訓練資料: 使用課堂所提供的40筆smoker data進行訓練，分別為10筆Current Smoker，10筆Non Smoker，10筆Past Smoker以及10筆Unknown的資料。

2. 測試資料: 同樣使用課堂所提供的40筆未提供ground truth的smoker data進行測試。

3. 程式使用:
  (1) 首先將word_frequency.py與smoker data都放置於同一個資料夾，並在py檔中設置欲篩選出的單字以及要將哪些單字的frequency加總合併計算並輸出至文字檔案。
  (2) 將輸出檔案檔名設置為test_data.txt，與Train.py放置於同一個資料夾使用。
  (3) 步驟(1)中欲使用的單字可以使用K_means_test.ipynb進行測試。

4. 於result_decision_tree.txt中可得到單純使用decision tree進行分類所得到的結果。

5. 於result_random_forest.txt中得到使用random forest進行分類得到的結果。