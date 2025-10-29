## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
     import pandas as pd 
     df= pd.read_csv("/content/Encoding Data.csv")
     df

   <img width="231" height="246" alt="image" src="https://github.com/user-attachments/assets/d9c4f7f8-fce7-447e-879a-9a7e997b87f7" />
   
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm= ['Hot','Warm','Cold']
e1= OrdinalEncoder (categories=[pm])
e1.fit_transform (df[["ord_2"]])

<img width="484" height="468" alt="image" src="https://github.com/user-attachments/assets/28aeac77-37e8-44a7-b417-9d5e3f7dce48" />

df['bo2']= e1.fit_transform(df[["ord_2"]])
df

<img width="479" height="444" alt="image" src="https://github.com/user-attachments/assets/fd0868bf-d0c0-440c-8753-f528a6624bf5" />

le= LabelEncoder()
dfc= df.copy()
dfc['ord_2']=le.fit_transform (dfc['ord_2'])
dfc

<img width="643" height="454" alt="image" src="https://github.com/user-attachments/assets/f1c0720f-0734-4de5-82c1-425f238d59e2" />

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2= pd.concat([df2,enc],axis=1)
df2

<img width="903" height="469" alt="image" src="https://github.com/user-attachments/assets/38e751c7-a566-4fa5-9dfe-e38c9124f256" />

pd.get_dummies(df2,columns=["nom_0"])


<img width="679" height="464" alt="image" src="https://github.com/user-attachments/assets/b8151012-6de4-424e-a1f8-46742d981195" />
from category_encoders import BinaryEncoder
df= pd.read_csv("/content/data.csv")
df

<img width="688" height="512" alt="image" src="https://github.com/user-attachments/assets/8512583d-7bc1-4abb-90aa-53aa29ef4ad6" />
be= BinaryEncoder()
nd= be.fit_transform(df['Ord_2'])
df

<img width="959" height="469" alt="image" src="https://github.com/user-attachments/assets/5c26002f-f352-4895-ab74-89335d111548" />
dfb= pd.concat([df,nd],axis=1)
dfb

<img width="768" height="471" alt="image" src="https://github.com/user-attachments/assets/a8c68f6f-b1ed-4b19-a304-ff7016a57a77" />
from category_encoders import TargetEncoder
te= TargetEncoder()
CC= df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC= pd.concat([CC,new],axis=1)
CC

<img width="1064" height="507" alt="image" src="https://github.com/user-attachments/assets/4af22c98-7813-41c9-a733-e743a92e7660" />
import pandas as pd 
import numpy as np
from scipy import stats 
df= pd.read_csv("/content/Data_to_Transform.csv")
df

<img width="437" height="246" alt="image" src="https://github.com/user-attachments/assets/42e2443c-b2c1-4e18-acce-948409e31e5a" />
df.skew()

<img width="407" height="564" alt="image" src="https://github.com/user-attachments/assets/3c23ea02-ed58-4077-b81e-89b4e1bfd68f" />
np.log(df["Highly Positive Skew"])

<img width="408" height="511" alt="image" src="https://github.com/user-attachments/assets/0805b0d9-61e0-4af1-baf6-d944002035ea" />
np.reciprocal(df["Moderate Positive Skew"])

<img width="361" height="518" alt="image" src="https://github.com/user-attachments/assets/7d4ef7c9-72ef-488b-b9ca-e256efbbb159" />
np.sqrt(df["Highly Positive Skew"])

 <img width="387" height="520" alt="image" src="https://github.com/user-attachments/assets/01ac15bc-c04e-44e8-888f-c15478b8648d" />
 np.square(df["Highly Positive Skew"])

 <img width="1382" height="524" alt="image" src="https://github.com/user-attachments/assets/3f07e0c8-872e-4dab-80b9-c65ac254a933" />
df["Highly Positive Skew_boxcox"],parameters= stats.boxcox(df["Highly Positive Skew"])
df

<img width="484" height="301" alt="image" src="https://github.com/user-attachments/assets/ccf5cf08-156b-45d7-a853-d213957e43df" />
df.skew()

<img width="536" height="353" alt="image" src="https://github.com/user-attachments/assets/26f43f51-2070-41c2-a4b6-31f009de3f10" />
df["Highly Negative Skew_yeojohnson"],parameters= stats.yeojohnson(df["Highly Negative Skew"])
df.skew()

<img width="1453" height="586" alt="image" src="https://github.com/user-attachments/assets/a00a8955-7cce-4847-b6c1-dea5a3ce0395" />
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate  Negative Skew"]])
 df

 <img width="773" height="556" alt="image" src="https://github.com/user-attachments/assets/c9105489-799e-4bea-84e3-627430f7ecb1" />
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

<img width="747" height="524" alt="image" src="https://github.com/user-attachments/assets/819e11e4-04d1-488a-a1b2-ec76e3dc4e88" />

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

<img width="708" height="541" alt="image" src="https://github.com/user-attachments/assets/be39a1bb-259b-40ca-a0d5-1c7a3321b56b" />

 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()

<img width="739" height="550" alt="image" src="https://github.com/user-attachments/assets/ff175685-252c-4354-98d8-774c952429dc" />

 df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()

 <img width="708" height="523" alt="image" src="https://github.com/user-attachments/assets/5bff67a6-9f19-4aed-afcc-3f4ef7209dbe" />
 dt =pd.read_csv("titanic_dataset.csv")
dt
dt=pd.read_csv("titanic_dataset.csv")
dt
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()

<img width="706" height="527" alt="image" src="https://github.com/user-attachments/assets/01dfc4c8-63d9-4534-a23e-722328599129" />


sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()


![Uploading image.png…]()


# RESULT:
             Thus the given data, Feature Encoding, Transformation process and save the data to a file  was performed successfully

       
