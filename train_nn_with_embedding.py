from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# from lime.lime_tabular import LimeTabularExplainer
from lime_tabular import LimeTabularExplainer
from models import NN_with_EntityEmbedding, NN
from data_script import get_data

run_id = 0
filtered_tasks_ids = [3, 20, 24, 41, 45, 49, 3492, 3493, 3494, 3560, 34537, 34539, 146195]
epochs = 50
batch_size = 128

for j in range(len(filtered_tasks_ids)):
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    print("++++++++++ tasks: "+str(j+1)+"/"+str(len(filtered_tasks_ids))+"+++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    # within different runs, split traning/testing sets with different
    # random_state.
    taskid = filtered_tasks_ids[j]
    X, X_train, X_test, y_train_str, y_test_str, y_train_int, y_test_int, feature_names, class_names, categorical_names, categorical_features = get_data(taskid, random_state=run_id)

    # nn with embedding related
    nn_with_embedding = NN_with_EntityEmbedding(X_train, y_train_int,
                                                categorical_features,
                                                categorical_names,
                                                class_names,
                                                epochs=epochs,
                                                batch_size=batch_size,
    )
    nn_with_embedding_loss, nn_with_embedding_score = nn_with_embedding.evaluate(X_test, y_test_int)
    print("nn_with_embedding prediction score: ", str(nn_with_embedding_score))

    weight_path = '/content/drive/My Drive/myenv/model_weight_'+str(j)+'.h5'
    architecture_path = '/content/drive/My Drive/myenv/model_architecture_'+str(j)+'.json'

    # Save the weights
    nn_with_embedding.model.save_weights(weight_path)
    # Save the model architecture
    with open(architecture_path, 'w') as f:
        f.write(nn_with_embedding.model.to_json())

