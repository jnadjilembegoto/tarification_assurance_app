import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# module python utilisé pour enregistrer le modèle ML
import joblib 

# Fonction de prétraitement des données
def treatInput(data):
    try:
      
        # Créer la variable ClaimFreq qui définit la fréquence de sinistre par période d'exposition
        data["ClaimFreq"] = data["ClaimNb"] / data["Exposure"]

        # encoder les données catégorielles

        # Effectuons une copie des données d'apprentissage pour éviter de les modifier
        input_copy = data.copy()

        # Appliquons le label encoder à la colonne Power( transformer les valeurs nominales en valeurs numériques)
        label_encoder = LabelEncoder()

        input_copy['Power'] = label_encoder.fit_transform(data['Power'])

        # print(input_copy['Power'])
        print("here treatInput")
        # Initialiser le one-hot encoder pour les colonnes Brand, Gas et Region
        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 
        # handle_unknow = 'ignore': Lorsqu'une catégorie inconnue est rencontrée au cours de la transformation, les colonnes codées à un point qui en résultent pour cette caractéristique seront toutes des zéros.
        # sparse = False : pour s'assurer que les données encodées soient un tableau numpy
        OH_cols = ['Brand', 'Gas', 'Region']

        # Transformation
        OH_cols_treat = pd.DataFrame(OH_encoder.fit_transform(input_copy[OH_cols]))
        
        # print(OH_cols_treat)

        # Réassigner l'index original car l'encodage One-hot supprime l'index
        OH_cols_treat.index = input_copy.index

        # Remettre l'étiquetage des colonnes à l'aide de la fonction get_feature_names_out(). 
        OH_cols_treat.columns = OH_encoder.get_feature_names_out(OH_cols)

        # Créer une copie des données numériques qui n'ont pas été encodées
        input_copy_no_OH_cols = input_copy.drop(OH_cols, axis=1)
    

        # Concaténer à présent les données encodées avec celles qui sont non encodées
        input_copy_enc = pd.concat([input_copy_no_OH_cols, OH_cols_treat], axis=1)
        print("data_encoded",input_copy_enc)
        st.write("données encodées")

        if input_copy_enc["Gas_Regular"].iloc[0] == 1.0:
            input_copy_enc = input_copy_enc.drop("Gas_Regular", axis=1)
            input_copy_enc["Gas_Diesel"] = 0.0

        # training data columns
        training_columns = ['ClaimNb', 'Exposure', 'Power', 'CarAge', 'DriverAge', 'Density',
       'ClaimFreq', 'Brand_Fiat', 'Brand_Japanese (except Nissan) or Korean',
       'Brand_Mercedes, Chrysler or BMW', 'Brand_Opel, General Motors or Ford',
       'Brand_Renault, Nissan or Citroen',
       'Brand_Volkswagen, Audi, Skoda or Seat', 'Brand_other', 'Gas_Diesel',
       'Region_Aquitaine', 'Region_Basse-Normandie', 'Region_Bretagne',
       'Region_Centre', 'Region_Haute-Normandie', 'Region_Ile-de-France',
       'Region_Limousin', 'Region_Nord-Pas-de-Calais',
       'Region_Pays-de-la-Loire', 'Region_Poitou-Charentes']
        
        # rajouter les colonnes manquantes dans le jeu de données
        for col in training_columns:
            if col not in input_copy_enc.columns:
                input_copy_enc[col] = 0.0
        

        # Réorganiser les colonnes dans le bon ordre
        input_copy_enc = input_copy_enc[training_columns]

        return input_copy_enc
    except:
        exit()
    
   

# Fonction qui gère la prédiction de la prime
def predictPrime(data):
    print(data)
    try:
        # Chargement du modèle pré-entraîné C:\Users\EEIA\Desktop\Assurance_project\tweedieModel.pkl
        with open('tweedieModel.pkl', 'rb') as file:
            print(file)
            model = joblib.load(file)
            st.write("model chargé")
            print("model chargé",model)
            prediction = model.predict(data)
            print("prediction result",prediction)
            if prediction:
                return prediction[0]
    except:
        exit()

# Configuration de la page
st.set_page_config(page_title="Caractéristiques de l'Assuré et de l'Automobile", page_icon=":car:")

st.title("Formulaire de Caractéristiques de l'Assuré et de l'Automobile")

# Section 1 : Caractéristiques de l'Assuré
st.header("Caractéristiques de l'Assuré")

age = st.number_input("Âge de l'assuré", min_value=18, max_value=100, step=1, help="Entrez l'âge de l'assuré.")
identifiant_contrat = st.text_input("Identifiant du contrat", help="Entrez l'identifiant unique du contrat.")
periode_couverture = st.date_input(
    "Période de couverture de l'assurance",
    value=[date.today(), date.today().replace(year=date.today().year + 1)],
    help="Sélectionnez la période de début et de fin de la couverture."
)
region_habitation = st.selectbox(
    "Région d'habitation",
    ['Aquitaine', 'Nord-Pas-de-Calais', 'Pays-de-la-Loire',
       'Ile-de-France', 'Centre', 'Poitou-Charentes', 'Bretagne',
       'Basse-Normandie', 'Limousin', 'Haute-Normandie'],
    help="Sélectionnez la région d'habitation de l'assuré."
)
densite_population = st.slider(
    "Densité de la région (habitants par km²)",
    min_value=10, max_value=10000, step=10,
    help="Entrez la densité de la population dans la région d'habitation de l'assuré."
)

# Section 2 : Caractéristiques de l'Automobile
st.header("Caractéristiques de l'Automobile")

marque = st.selectbox(
    "Marque de l'automobile",
    ['Japanese (except Nissan) or Korean', 'Fiat', 'Opel, General Motors or Ford', 'Mercedes, Chrysler or BMW',
'Renault, Nissan or Citroen', 'Volkswagen, Audi, Skoda or Seat','other'],
    help="Sélectionnez la marque de l'automobile."
)
# puissance = st.number_input(
#     "Puissance de l'automobile (en chevaux)",
#     min_value=50, max_value=1000, step=10,
#     help="Entrez la puissance de l'automobile en chevaux."
# )
puissance = st.selectbox(
    "Puissance de l'automobile",
    ['d', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'],
    help="Entrez la puissance de l'automobile."
)
age_vehicule = st.number_input(
    "Âge de l'automobile (en années)",
    min_value=0, max_value=50, step=1,
    help="Entrez l'âge du véhicule en années."
)
carburation = st.radio(
    "Type de carburation",
    ["Regular", "Diesel"],
    help="Sélectionnez le type de carburation de l'automobile. Regular(Essence, Electrique, Hybride)"
)

# Section 3 : Sinistres survenus
st.header("Sinistres survenus")

nombre_sinistres = st.number_input(
    "Nombre de sinistres survenus",
    min_value=0, max_value=100, step=1,
    help="Entrez le nombre de sinistres survenus durant la période de couverture."
)

# Bouton de soumission
if st.button("Soumettre les informations"):

    st.success("Les informations ont été soumises avec succès!")
    
    # Enregistrer les données sous forme de dictionnaire
    input_data = {
        'DriverAge': age,
        'Exposure': ((periode_couverture[1] - periode_couverture[0]).days)/365.0,  # Période en jours / 365.0
        'Region': region_habitation,
        'Density': densite_population,
        'Brand': marque,
        'Power': puissance,
        'CarAge': age_vehicule,
        'Gas': carburation,
        'ClaimNb': nombre_sinistres 
    }

    # Transformation des données en DataFrame
    input_df = pd.DataFrame([input_data])

    # Prétraiter les données avant de les passer au modèle
    input_treated = treatInput(input_df)

    print("données encodées", input_treated)

    # Calcul de la prime
    prime = predictPrime(input_treated)

    # Exemple d'affichage des données saisies :
    st.write("## Résumé des informations saisies :")
    st.write(f"**Âge de l'assuré :** {age} ans")
    st.write(f"**Identifiant du contrat :** {identifiant_contrat}")
    st.write(f"**Période de couverture :** du {periode_couverture[0]} au {periode_couverture[1]}")
    st.write(f"**Région d'habitation :** {region_habitation}")
    st.write(f"**Densité de population :** {densite_population} habitants/km²")
    st.write(f"**Marque de l'automobile :** {marque}")
    st.write(f"**Puissance de l'automobile :** {puissance} chevaux")
    st.write(f"**Âge de l'automobile :** {age_vehicule} ans")
    st.write(f"**Type de carburation :** {carburation}")
    st.write(f"**Nombre de sinistres survenus :** {nombre_sinistres}")

    # Affichage du résultat de la prédiction
    st.write("### Estimation de la Prime d'Assurance :")
    st.write(f"La prime d'assurance estimée pour cet assuré est de : **{prime:.2f} €**")
