# 台股 AI Scanner v1

這是一套台股盤後 AI 掃描系統。

功能：
- 自動抓取台股日資料
- 計算技術指標
- 加入三大法人買賣超
- 加入融資融券變化
- 產出 AI 總分、技術分、籌碼分、風險分
- 顯示今日 AI 排名
- 顯示單檔 K 線、均線與成交量
- 提供進場、出場與風險參考

執行方式：

```bash
pip install -r requirements.txt
streamlit run app.py
---

# 第 4 步：打包成 ZIP

新增一格，貼：

```python
!zip -r TW_Stock_AI_Scanner_v1.zip app.py requirements.txt README.md
