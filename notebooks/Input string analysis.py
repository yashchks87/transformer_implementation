# Databricks notebook source
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt
pd.set_option('display.float_format', '{:.2f}'.format)

# COMMAND ----------

main_csv = pd.read_csv('/Volumes/daai_ke_team/default/mfg_ai_images/translate_data/en-fr.csv')

# COMMAND ----------

main_csv.head()

# COMMAND ----------

main_csv['en_length'] = main_csv['en'].progress_apply(lambda x: len(str(x).split()))

# COMMAND ----------

main_csv.count()

# COMMAND ----------

main_csv[main_csv['en_length'] < 30].count()

# COMMAND ----------

22520376 - 16052476

# COMMAND ----------

main_csv['en_length'].describe()

# COMMAND ----------


