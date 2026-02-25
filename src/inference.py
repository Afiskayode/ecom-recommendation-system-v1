import os
import pickle


def load_model():

    with open('data/item_user_matrix.pkl', 'rb') as file:
        matrix = pickle.load(file)

    with open('data/model_knn.pkl', 'rb') as file:
        model = pickle.load(file)

    return model, matrix

model, item_user_matrix = load_model()

def get_recommendations(product_id: str, n_recommendations = 6):
    try:

        if product_id not in item_user_matrix.index:
            return None
         
        distances, indices = model.kneighbors(item_user_matrix.loc[product_id, :].values.reshape(1, -1), \
                            n_neighbors = n_recommendations)
        
        recommendations = []

        for i in range(1, len(distances.flatten())):
            rec_id = item_user_matrix.index[indices.flatten()[i]]
            recommendations.append({"product_id":rec_id, "distance":float(distances.flatten()[i])})

        return recommendations
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return []
    


