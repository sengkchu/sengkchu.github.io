---
title: Building RAG Systems on Databricks Part 1
date: 2025-07-02 00:00:00 +/-0000
categories: [AI, RAG System]
tags: [databricks, langchain, rag]
---

```python
%pip install langchain==0.1.5
dbutils.library.restartPython()
```


```python
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import GlobalConfig

cfg = GlobalConfig()
```


```python
# Read in data
data_path = f"/Volumes/{cfg.catalog}/{cfg.schema}/{cfg.volume_name}/{cfg.source_file_name}"
df = spark.read.text(data_path)

# Collect all the text into a single string
text_column = " ".join([row.value for row in df.collect()])

# Chunk out text
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=cfg.chunk_size,
    chunk_overlap=cfg.chunk_overlap,
    length_function=len,
)
chunks = splitter.split_text(text_column)
```


```python
chunked_pdf = pd.DataFrame({
    cfg.primary_key: range(1, len(chunks)+1),
    cfg.text_column: chunks
})
chunked_df = spark.createDataFrame(chunked_pdf)
chunked_df.write.mode("overwrite").saveAsTable(f"{cfg.catalog}.{cfg.schema}.{cfg.text_table_name}")
```


```python
query = f"""
ALTER TABLE {cfg.catalog}.{cfg.schema}.{cfg.text_table_name}
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
"""

spark.sql(query)
```
