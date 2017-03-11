
# In[1]:

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# In[2]:

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[3]:

train.shape


# Out[3]:

#     (8523, 12)

# In[4]:

test.shape


# Out[4]:

#     (5681, 11)

# In[8]:

train["source"] = "train"
test["source"] = "test"
data = pd.concat([train, test],ignore_index=True)
data.shape


# Out[8]:

#     (14204, 13)

# In[9]:

data.head()


# Out[9]:

#       Item_Fat_Content Item_Identifier  Item_MRP  Item_Outlet_Sales  \
#     0          Low Fat           FDA15  249.8092          3735.1380   
#     1          Regular           DRC01   48.2692           443.4228   
#     2          Low Fat           FDN15  141.6180          2097.2700   
#     3          Regular           FDX07  182.0950           732.3800   
#     4          Low Fat           NCD19   53.8614           994.7052   
#     
#                    Item_Type  Item_Visibility  Item_Weight  \
#     0                  Dairy         0.016047         9.30   
#     1            Soft Drinks         0.019278         5.92   
#     2                   Meat         0.016760        17.50   
#     3  Fruits and Vegetables         0.000000        19.20   
#     4              Household         0.000000         8.93   
#     
#        Outlet_Establishment_Year Outlet_Identifier Outlet_Location_Type  \
#     0                       1999            OUT049               Tier 1   
#     1                       2009            OUT018               Tier 3   
#     2                       1999            OUT049               Tier 1   
#     3                       1998            OUT010               Tier 3   
#     4                       1987            OUT013               Tier 3   
#     
#       Outlet_Size        Outlet_Type source  
#     0      Medium  Supermarket Type1  train  
#     1      Medium  Supermarket Type2  train  
#     2      Medium  Supermarket Type1  train  
#     3         NaN      Grocery Store  train  
#     4        High  Supermarket Type1  train  
#     
#     [5 rows x 13 columns]

# In[14]:

data.apply(lambda x: sum(x.isnull()))


# Out[14]:

#     Item_Fat_Content                0
#     Item_Identifier                 0
#     Item_MRP                        0
#     Item_Outlet_Sales            5681
#     Item_Type                       0
#     Item_Visibility                 0
#     Item_Weight                  2439
#     Outlet_Establishment_Year       0
#     Outlet_Identifier               0
#     Outlet_Location_Type            0
#     Outlet_Size                  4016
#     Outlet_Type                     0
#     source                          0
#     dtype: int64

# In[15]:

data.describe()


# Out[15]:

#                Item_MRP  Item_Outlet_Sales  Item_Visibility   Item_Weight  \
#     count  14204.000000        8523.000000     14204.000000  11765.000000   
#     mean     141.004977        2181.288914         0.065953     12.792854   
#     std       62.086938        1706.499616         0.051459      4.652502   
#     min       31.290000          33.290000         0.000000      4.555000   
#     25%       94.012000         834.247400         0.027036      8.710000   
#     50%      142.247000        1794.331000         0.054021     12.600000   
#     75%      185.855600        3101.296400         0.094037     16.750000   
#     max      266.888400       13086.964800         0.328391     21.350000   
#     
#            Outlet_Establishment_Year  
#     count               14204.000000  
#     mean                 1997.830681  
#     std                     8.371664  
#     min                  1985.000000  
#     25%                  1987.000000  
#     50%                  1999.000000  
#     75%                  2004.000000  
#     max                  2009.000000  
#     
#     [8 rows x 5 columns]

# In[18]:

data.apply(lambda x: len(x.unique()),axis=0)


# Out[18]:

#     Item_Fat_Content                 5
#     Item_Identifier               1559
#     Item_MRP                      8052
#     Item_Outlet_Sales             3494
#     Item_Type                       16
#     Item_Visibility              13006
#     Item_Weight                    416
#     Outlet_Establishment_Year        9
#     Outlet_Identifier               10
#     Outlet_Location_Type             3
#     Outlet_Size                      4
#     Outlet_Type                      4
#     source                           2
#     dtype: int64

# In[19]:

data.dtypes


# Out[19]:

#     Item_Fat_Content              object
#     Item_Identifier               object
#     Item_MRP                     float64
#     Item_Outlet_Sales            float64
#     Item_Type                     object
#     Item_Visibility              float64
#     Item_Weight                  float64
#     Outlet_Establishment_Year      int64
#     Outlet_Identifier             object
#     Outlet_Location_Type          object
#     Outlet_Size                   object
#     Outlet_Type                   object
#     source                        object
#     dtype: object

# In[20]:

data.dtypes.index


# Out[20]:

#     Index([u'Item_Fat_Content', u'Item_Identifier', u'Item_MRP', u'Item_Outlet_Sales', u'Item_Type', u'Item_Visibility', u'Item_Weight', u'Outlet_Establishment_Year', u'Outlet_Identifier', u'Outlet_Location_Type', u'Outlet_Size', u'Outlet_Type', u'source'], dtype='object')

# In[25]:

categorical_columns = [i for i in data.columns if data.dtypes[i]=="object"]


# In[26]:

categorical_columns


# Out[26]:

#     ['Item_Fat_Content',
#      'Item_Identifier',
#      'Item_Type',
#      'Outlet_Identifier',
#      'Outlet_Location_Type',
#      'Outlet_Size',
#      'Outlet_Type',
#      'source']

# In[28]:

categorical_columns = [i for i in data.columns if data.dtypes[i]=="object"]


# categorical_columns

# In[29]:

categorical_columns


# Out[29]:

#     ['Item_Fat_Content',
#      'Item_Identifier',
#      'Item_Type',
#      'Outlet_Identifier',
#      'Outlet_Location_Type',
#      'Outlet_Size',
#      'Outlet_Type',
#      'source']

# In[30]:

categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]


# In[31]:

categorical_columns


# Out[31]:

#     ['Item_Fat_Content',
#      'Item_Type',
#      'Outlet_Location_Type',
#      'Outlet_Size',
#      'Outlet_Type']

# In[33]:

for c in categorical_columns:
    print "column name is ", c
    print data[c].value_counts()


# Out[33]:

#     column name is  Item_Fat_Content
#     Low Fat    8485
#     Regular    4824
#     LF          522
#     reg         195
#     low fat     178
#     dtype: int64
#     column name is  Item_Type
#     Fruits and Vegetables    2013
#     Snack Foods              1989
#     Household                1548
#     Frozen Foods             1426
#     Dairy                    1136
#     Baking Goods             1086
#     Canned                   1084
#     Health and Hygiene        858
#     Meat                      736
#     Soft Drinks               726
#     Breads                    416
#     Hard Drinks               362
#     Others                    280
#     Starchy Foods             269
#     Breakfast                 186
#     Seafood                    89
#     dtype: int64
#     column name is  Outlet_Location_Type
#     Tier 3    5583
#     Tier 2    4641
#     Tier 1    3980
#     dtype: int64
#     column name is  Outlet_Size
#     Medium    4655
#     Small     3980
#     High      1553
#     dtype: int64
#     column name is  Outlet_Type
#     Supermarket Type1    9294
#     Grocery Store        1805
#     Supermarket Type3    1559
#     Supermarket Type2    1546
#     dtype: int64
# 

# In[34]:

data['Item_Identifier'].value_counts()


# Out[34]:

#     NCK18    10
#     FDL46    10
#     DRH39    10
#     FDX50    10
#     FDO01    10
#     FDL40    10
#     FDG10    10
#     NCG18    10
#     FDS25    10
#     FDO08    10
#     NCF54    10
#     DRL49    10
#     FDT13    10
#     NCC06    10
#     FDG33    10
#     ...
#     FDG14    8
#     DRB24    8
#     FDR51    7
#     FDM10    7
#     FDI46    7
#     FDM50    7
#     FDM52    7
#     NCW54    7
#     NCL42    7
#     FDH58    7
#     FDS22    7
#     FDX49    7
#     DRN11    7
#     FDO33    7
#     FDL50    7
#     Length: 1559, dtype: int64

# In[35]:

item_avg_weight = data.pivot_table(values="Item_Weight", index="Item_Identifier")


# Out[35]:


    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    <ipython-input-35-41f317a94440> in <module>()
    ----> 1 item_avg_weight = data.pivot_table(values="Item_Weight",index="Item_Identifier")
    

    TypeError: pivot_table() got an unexpected keyword argument 'index'


# In[38]:

item_avg_weight = data.pivot_table(values='Item_Weight', index=['Item_Identifier'])


# Out[38]:


    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    <ipython-input-38-ca72684c199f> in <module>()
    ----> 1 item_avg_weight = data.pivot_table(values='Item_Weight', index=['Item_Identifier'])
    

    TypeError: pivot_table() got an unexpected keyword argument 'index'


# In[39]:

item_avg_weight = data.pivot_table(values='Item_Weight', index=['Item_Identifier'])


# Out[39]:


    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    <ipython-input-39-ca72684c199f> in <module>()
    ----> 1 item_avg_weight = data.pivot_table(values='Item_Weight', index=['Item_Identifier'])
    

    TypeError: pivot_table() got an unexpected keyword argument 'index'


# In[40]:

item_avg_weight = pivot_table(data, values='Item_Weight', index=['Item_Identifier'])


# Out[40]:


    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)

    <ipython-input-40-6eaf7ac0d63f> in <module>()
    ----> 1 item_avg_weight = pivot_table(data, values='Item_Weight', index=['Item_Identifier'])
    

    NameError: name 'pivot_table' is not defined


# In[41]:

item_avg_weight = pd.pivot_table(data, values='Item_Weight', index=['Item_Identifier'])


# Out[41]:


    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    <ipython-input-41-84394325f4db> in <module>()
    ----> 1 item_avg_weight = pd.pivot_table(data, values='Item_Weight', index=['Item_Identifier'])
    

    TypeError: pivot_table() got an unexpected keyword argument 'index'


# In[42]:

item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')


# Out[42]:


    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    <ipython-input-42-1ec67e4a5afd> in <module>()
    ----> 1 item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')
    

    TypeError: pivot_table() got an unexpected keyword argument 'index'


# In[43]:

item_avg_weight = data.pivot(values='Item_Weight', index='Item_Identifier')


# Out[43]:


    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)

    <ipython-input-43-6289cc1d12c6> in <module>()
    ----> 1 item_avg_weight = data.pivot(values='Item_Weight', index='Item_Identifier')
    

    /usr/lib/python2.7/dist-packages/pandas/core/frame.pyc in pivot(self, index, columns, values)
       3114         """
       3115         from pandas.core.reshape import pivot
    -> 3116         return pivot(self, index=index, columns=columns, values=values)
       3117 
       3118     def stack(self, level=-1, dropna=True):


    /usr/lib/python2.7/dist-packages/pandas/core/reshape.pyc in pivot(self, index, columns, values)
        349         indexed = Series(self[values].values,
        350                          index=MultiIndex.from_arrays([self[index],
    --> 351                                                        self[columns]]))
        352         return indexed.unstack(columns)
        353 


    /usr/lib/python2.7/dist-packages/pandas/core/frame.pyc in __getitem__(self, key)
       1656             return self._getitem_multilevel(key)
       1657         else:
    -> 1658             return self._getitem_column(key)
       1659 
       1660     def _getitem_column(self, key):


    /usr/lib/python2.7/dist-packages/pandas/core/frame.pyc in _getitem_column(self, key)
       1663         # get column
       1664         if self.columns.is_unique:
    -> 1665             return self._get_item_cache(key)
       1666 
       1667         # duplicate columns & possible reduce dimensionaility


    /usr/lib/python2.7/dist-packages/pandas/core/generic.pyc in _get_item_cache(self, item)
       1003         res = cache.get(item)
       1004         if res is None:
    -> 1005             values = self._data.get(item)
       1006             res = self._box_item_values(item, values)
       1007             cache[item] = res


    /usr/lib/python2.7/dist-packages/pandas/core/internals.pyc in get(self, item)
       2870             if isnull(item):
       2871                 indexer = np.arange(len(self.items))[isnull(self.items)]
    -> 2872                 return self.get_for_nan_indexer(indexer)
       2873 
       2874             _, block = self._find_block(item)


    /usr/lib/python2.7/dist-packages/pandas/core/internals.pyc in get_for_nan_indexer(self, indexer)
       2922                 indexer = indexer.item()
       2923             else:
    -> 2924                 raise ValueError("cannot label index with a null key")
       2925 
       2926         # take a nan indexer and return the values


    ValueError: cannot label index with a null key


# In[44]:

df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 6, 'B': ['A', 'B', 'C'] * 8,'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 4,'D': np.random.randn(24),'E': np.random.randn(24),'F': [datetime.datetime(2013, i, 1) for i in range(1, 13)] +[datetime.datetime(2013, i, 15) for i in range(1, 13)]})
df


# Out[44]:


    ---------------------------------------------------------------------------
    NameError                                 Traceback (most recent call last)

    <ipython-input-44-8036c637e86f> in <module>()
    ----> 1 df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 6, 'B': ['A', 'B', 'C'] * 8,'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 4,'D': np.random.randn(24),'E': np.random.randn(24),'F': [datetime.datetime(2013, i, 1) for i in range(1, 13)] +[datetime.datetime(2013, i, 15) for i in range(1, 13)]})
          2 df


    NameError: name 'datetime' is not defined


# In[45]:

import datetime


# In[46]:

df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 6, 'B': ['A', 'B', 'C'] * 8,'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 4,'D': np.random.randn(24),'E': np.random.randn(24),'F': [datetime.datetime(2013, i, 1) for i in range(1, 13)] +[datetime.datetime(2013, i, 15) for i in range(1, 13)]})


# In[47]:

df


# Out[47]:

#             A  B    C         D         E          F
#     0     one  A  foo -0.043919  0.352704 2013-01-01
#     1     one  B  foo  0.202232  0.591719 2013-02-01
#     2     two  C  foo -1.092058  0.612976 2013-03-01
#     3   three  A  bar  0.538749 -0.050952 2013-04-01
#     4     one  B  bar  0.061395  1.322045 2013-05-01
#     5     one  C  bar -0.146826 -0.935600 2013-06-01
#     6     two  A  foo  1.212213  0.134534 2013-07-01
#     7   three  B  foo -2.019199  2.057265 2013-08-01
#     8     one  C  foo -0.670599  0.278305 2013-09-01
#     9     one  A  bar  1.238409 -0.289759 2013-10-01
#     10    two  B  bar -0.106110  0.517796 2013-11-01
#     11  three  C  bar -0.403674  0.851759 2013-12-01
#     12    one  A  foo -1.637121 -1.430462 2013-01-15
#     13    one  B  foo  1.087702  1.684226 2013-02-15
#     14    two  C  foo -0.862232 -1.322353 2013-03-15
#     15  three  A  bar -1.999387  1.255191 2013-04-15
#     16    one  B  bar  1.761171 -1.185200 2013-05-15
#     17    one  C  bar -0.794793  0.553544 2013-06-15
#     18    two  A  foo  0.716997 -0.509581 2013-07-15
#     19  three  B  foo  0.975265  0.345160 2013-08-15
#     20    one  C  foo  1.066313  1.886262 2013-09-15
#     21    one  A  bar -1.534781  0.488611 2013-10-15
#     22    two  B  bar -1.732207  0.686126 2013-11-15
#     23  three  C  bar  2.234340 -1.015567 2013-12-15
#     
#     [24 rows x 6 columns]

# In[48]:

pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])


# Out[48]:


    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    <ipython-input-48-e89ea23b115f> in <module>()
    ----> 1 pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
    

    TypeError: pivot_table() got an unexpected keyword argument 'index'


# In[56]:

pd.__version__


# Out[56]:

#     '0.13.1'

# In[50]:

miss_bool = data['Item_Weight'].isnull() 


# In[53]:

sum(miss_bool)


# Out[53]:

#     2439

# In[54]:

type(miss_bool)


# Out[54]:

#     pandas.core.series.Series

# In[55]:

pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])


# Out[55]:


    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    <ipython-input-55-e89ea23b115f> in <module>()
    ----> 1 pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
    

    TypeError: pivot_table() got an unexpected keyword argument 'index'


# In[57]:

import pandas as pd


# In[58]:

pd.__version__


# Out[58]:

#     '0.13.1'

# In[59]:

from IPython.nbformat import current as nbformat
from IPython.nbconvert import PythonExporter


# Out[59]:


    ---------------------------------------------------------------------------
    ImportError                               Traceback (most recent call last)

    <ipython-input-59-954cc84c320b> in <module>()
          1 from IPython.nbformat import current as nbformat
    ----> 2 from IPython.nbconvert import PythonExporter
    

    /usr/lib/python2.7/dist-packages/IPython/nbconvert/__init__.py in <module>()
          1 """Utilities for converting notebooks to and from different formats."""
          2 
    ----> 3 from .exporters import *
          4 import filters
          5 import transformers


    /usr/lib/python2.7/dist-packages/IPython/nbconvert/exporters/__init__.py in <module>()
    ----> 1 from .export import *
          2 from .html import HTMLExporter
          3 from .slides import SlidesExporter
          4 from .exporter import Exporter
          5 from .latex import LatexExporter


    /usr/lib/python2.7/dist-packages/IPython/nbconvert/exporters/export.py in <module>()
         19 from IPython.config import Config
         20 
    ---> 21 from .exporter import Exporter
         22 from .html import HTMLExporter
         23 from .slides import SlidesExporter


    /usr/lib/python2.7/dist-packages/IPython/nbconvert/exporters/exporter.py in <module>()
         37 from IPython.utils import py3compat
         38 
    ---> 39 from IPython.nbconvert import transformers as nbtransformers
         40 from IPython.nbconvert import filters
         41 


    /usr/lib/python2.7/dist-packages/IPython/nbconvert/transformers/__init__.py in <module>()
          5 from .extractoutput import ExtractOutputTransformer
          6 from .revealhelp import RevealHelpTransformer
    ----> 7 from .latex import LatexTransformer
          8 from .sphinx import SphinxTransformer
          9 from .csshtmlheader import CSSHTMLHeaderTransformer


    /usr/lib/python2.7/dist-packages/IPython/nbconvert/transformers/latex.py in <module>()
         19 # Needed to override transformer
         20 from .base import (Transformer)
    ---> 21 from IPython.nbconvert import filters
         22 
         23 #-----------------------------------------------------------------------------


    /usr/lib/python2.7/dist-packages/IPython/nbconvert/filters/__init__.py in <module>()
          1 from .ansi import *
          2 from .datatypefilter import *
    ----> 3 from .highlight import *
          4 from .latex import *
          5 from .markdown import *


    /usr/lib/python2.7/dist-packages/IPython/nbconvert/filters/highlight.py in <module>()
         15 #-----------------------------------------------------------------------------
         16 
    ---> 17 from  pygments import highlight as pygements_highlight
         18 from pygments.lexers import get_lexer_by_name
         19 from pygments.formatters import HtmlFormatter


    ImportError: No module named pygments


# In[60]:

import nbformat
from nbconvert import PythonExporter


# Out[60]:


    ---------------------------------------------------------------------------
    ImportError                               Traceback (most recent call last)

    <ipython-input-60-f1e84d9913e7> in <module>()
    ----> 1 import nbformat
          2 from nbconvert import PythonExporter


    ImportError: No module named nbformat


# In[ ]:



