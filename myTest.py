import os
import numpy as np
import scipy
import scipy.misc
import pandas as pd
import matplotlib.pyplot as plt
import keras
dict_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson',
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel',
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lenny_leonard', 11:'lisa_simpson',
        12: 'marge_simpson', 13: 'mayor_quimby',14:'milhouse_van_houten', 15: 'moe_szyslak',
        16: 'ned_flanders', 17: 'nelson_muntz', 18: 'principal_skinner', 19: 'sideshow_bob'}





def load_ml_test(dirname):
    X_test = []
    for image_name in sorted(os.listdir(dirname),key=lambda x:int(x[:-4])):
        image = scipy.misc.imread(dirname+'/'+image_name)
        X_test.append(scipy.misc.imresize(image,(64,64),interp='lanczos'))
    return np.array(X_test)

X_test=load_ml_test("C:/Users/CaesarYu/Desktop/ML_HW02/data/test")
myfile=os.listdir("C:/Users/CaesarYu/Desktop/ML_HW02/data/test")
print(myfile)
print(sorted(myfile,key=lambda x:int(x[:-4])))
plt.imshow(X_test[1])
X_test=X_test/255.0
plt.show()
model = keras.models.load_model("best_model.hdf5")
# load weights into new model
Y_PREDICTION = model.predict(X_test)
Y_PREDICTION_classes = np.argmax(Y_PREDICTION, axis=1)
result_name=[]
for i in Y_PREDICTION_classes:
     result_name.append(dict_characters[i])
print(result_name)
result=pd.DataFrame(result_name)
result.index=np.arange(1,len(result_name)+1)
result.index.name='id'
result.columns=['character']
result.to_csv('OUTPUT1.csv')
# print(dict_characters[17])
#X_val_out, Y_val_out = load_test_set("C:/Users/CaesarYu/Desktop/ML_HW02/data/123456", dict_characters)
