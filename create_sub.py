# Creating the submission file \
def sub(x_val, test_df):

    model.load_weights('weights/model-ffn.hdf5')
    #X_test = np.array(x_val)
    predicted = model.predict(x_val)
    predicted
    ans_reg = predicted
    submit = pd.DataFrame()
    submit['Unique_ID'] = test_df['Unique_ID']
    submit['Views'] = ans_reg
    submit['Views'] = submit['Views'].apply(lambda x : int(x))
    file = "submission2"+".csv"
    submit.to_csv(file, index=False)
