#%%

from ps3.data._create_sample_column import create_sample_column

#%%

import os, sys

project_root = "/Users/louiserolland/cam_D100/ps3_claims"
os.chdir(project_root)
sys.path.insert(0, project_root)

print("NEW WORKING DIRECTORY:", os.getcwd())


#%%

from ps3.data import load_transform
#it is ps3.data not ps3_claims.data as the name of the package is ps3

#%%

df = load_transform()

df
#%%
df_split = create_sample_column(df, key_cols=["IDpol"], train_fraction=0.8)
df_split["sample"].value_counts()

df_split
# %%
