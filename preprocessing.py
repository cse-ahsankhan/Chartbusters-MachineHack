# Function to find out Unique Values Frequency
def UniqueValues(df):
    strg = "Number of unique values in "
    col = df.columns
    for i in col:
        print('-------------------------------------------------------------------')
        print(strg + "{} are \n{}".format(i, df[str(i)].nunique()))
        print('Unique Values in {} are \n{}'.format(i, df[str(i)].unique()))

#Function to perform LabelEncoding on the dataset
def LabelEncoding(df):
    enc = LabelEncoder()
    for i in df.columns:
        try:
            if np.dtype(df[str(i)]) == 'object' or np.dtype(df[str(i)]) == 'bool':
                df[str(i)] = enc.fit_transform(df[str(i)].astype(str))
        except:
            pass
    return df

# Function to preprocess the likes format of the given dataset i.e. to convert alpha numeric into float
def process_likes(x):
    x = x.replace(',', '')
    if x[-1] == 'K':
        return float(x[:-1])*1000
    if x[-1] == 'M':
        return float(x[:-1])*1000000
    if x[-1] == 'B':
        return float(x[:-1])*1000000000
    else:
        return int(x)

#Function to convert the datetime of the dataset in feasable format
def convert_time(x):
    date = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
    today = date.today()
    return (today - date).days
