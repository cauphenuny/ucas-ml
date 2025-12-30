== Logistic Regression

训练参数：

```python
make_pipeline(
    TfidfVectorizer(ngram_range=(1,2), max_features=50000),
    LogisticRegression(max_iter=2000, multi_class="multinomial", solver="saga"),
)
```

结果：Test Accuracy $=0.62450$，作为baseline

#image("assets/kaggle/traditional.png")

尝试了与深度学习方法集成，但效果不如直接使用深度学习模型。