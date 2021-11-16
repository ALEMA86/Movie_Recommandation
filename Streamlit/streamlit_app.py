######################################################################################
######################################################################################
###########################     LIBRAIRIES    ########################################
######################################################################################
######################################################################################

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import datetime as dt
import plotly.express as px
import ipywidgets as widgets
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors


######################################################################################
######################################################################################
###########################     DONNEES    ###########################################
######################################################################################
######################################################################################

df_recommandation = pd.read_csv('https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Database_projet/df_recommendation.csv?token=AU6BUZUA5UESEPKRRJQIESLBS53UU')
df = pd.read_csv('https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Database_projet/df_base.csv?token=AU6BUZWHN456IAMFBUWFFSDBTELCU')

######################################################################################
######################################################################################
###########################     FONCTIONS    #########################################
######################################################################################
######################################################################################

@st.cache
def load_df(url):
    df = pd.read_csv(url)
    df.set_index(df.iloc[:,0], inplace=True)
    df = df.iloc[:, 1:]
    return df





######################################################################################
######################################################################################
###########################     INTERFACE    #########################################
######################################################################################
######################################################################################


st.set_page_config( layout='wide')


def main():

    #st.title("Movie recommandation project")
    menu = ["Présentation du Projet", "Movie recommandation", "Meaningful KPI"]

    choice = st.sidebar.selectbox("Menu", menu) 

######################################################################################
######################################################################################
###########################     AURORE     ###########################################
######################################################################################
######################################################################################
    if choice == "Présentation du Projet":
        st.title("Présentation du Projet")
        st.subheader('')
        st.subheader("Le Projet")

        st.markdown(
        """
        Le _PROJET ABC'S_ est issu d’un projet d’école organisé par la __Wild Code School__. Il intervient dans le cadre de notre formation de Data Analyst, 2 mois après son début.

        L’objectif de ce projet est le suivant :

        Nous sommes une équipe de Data Analysts freelance.
        Un cinéma en perte de vitesse situé dans la Creuse nous contacte ca rl a décidé de passer le cap du digital en créant un site Internet taillé pour les locaux.
        Notre client nous demande de créer un moteur de recommandations de films qui à terme, enverra des notifications via internet.

        Aucun client du cinéma n'ayant à ce jour renseigné ses préférences, nous sommes donc dans une situation de __cold start__. Cependant, notre client nous a fourni une base de données basée sur la plateforme IMDb.

        """
        )
        st.subheader('')
        st.subheader("L'équipe")

        st.markdown(
        """
        Notre équipe est composée de 4 élèves issus de la promo Data Green de la __Wild Code School__ :
        - [Aurore LEMAÎTRE](https://github.com/alema86)
        - [Bérenger QUEUNE](https://github.com/BerengerQueune)
        - [Christophe LEFEBVRE](https://github.com/clefebvre2021)
        - [Stéphane ESSOUMAN](https://github.com/Liostephe)

        Tous les quatre formons l'équipe ABC's Data.
        """
        )
        col1, col2, col3 = st.columns(3)
        with col2:
            st.image("https://d1qg2exw9ypjcp.cloudfront.net/assets/prod/24134/210x210-9_cropped_1377120495_p182hcd8rofaq1t491u06kih16o13.png")

        st.subheader('')
        st.subheader("Notre client(e)")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("https://i.ibb.co/0hnKBMX/Framboise2.png")

        st.markdown(
        """
        Notre cliente est Framboise de Papincourt, petite fille du Comte de Montmirail. Elle a 25 ans et dirige un cinéma en perte de vitesse qui s'appelle "LE KINO".

        Elle fait appelle à nous car elle est désespérée. Son cinéma ne fait pas de bénéfice, ses créanciers sont à sa porte et ses problèmes financiers sont tels qu'elle a dû demandé un nouveau prêt dans une banque alors que c'est contre ses principes.

        Issue d'une famille de nobles, elle ne peut pas faire non plus appel à ses proches qui sont fortunés, car elle a renié sa famille. En effet ses derniers ne partagent pas sa vision des choses; exemple : elle est vegan alors que l'activité principale de sa famille est la chasse...

        Elle diffusait initialement des films qui la touchaient afin d'essayer de partager sa vision du monde. Ainsi, la films diffusés étaient principalement des documentaires traitant de l'écologie, du féminisme et de la paix universelle.

        Elle est obligée de faire changer de cap son cinéma et est prête à diffuser des films qui vont à l'encontre de ses convictions si ça lui permet de ne pas mettre la clé sous la porte et éviter d'être la raillerie de sa famille.
        Faire du bénéfice à terme serait un plus, car ça lui permettrait d'offrir à ses futurs enfants Harmony, Safran et Kiwi un environnement dans lequel ils pourront s'épanouir comme elle en rêve.

        Ainsi, elle nous donne carte blanche dans le rendu de notre travail.
        """  
        )    
        st.subheader('')
        st.subheader("Notre mission")

        st.markdown(
        """
        Nous devons fournir à notre client les outils d’analyse de la base de données issue de __IMDB__.

        Il est nécessaire de :
        - Faire une rapide présentation de la base de données (sur notre espace collaboratif sur Github)
        - Fournir à notre client quelques statistiques sur les films :
            - Films : types, durées...
            - Acteurs : nombre de films, type de films...
        - Présenter les TOP 10 des films par années et genre
        - Présenter les TOP 5 des acteurs/actrices par années et genre
        - Retourner une liste de films recommandés en fonction d'IDs ou de noms de films choisis par un utilisateur
        - Il faudra entraîner des outils de Machine Learning : 
	        - Recommandation de films proches d’un film cible grâce à un modèle de __KNN__
	        - Proposition d’une rétrospective avec un modèle de __Régression Logistique__
        """
        )

        st.subheader('')
        st.subheader("Outils")

        st.markdown(
        """
        Le projet est entièrement fait sous **Python**.

        Nous avons utilisés entres autres les librairies suivantes :    
        - Pandas
        - Sklearn
        - Plotly
        - Streamlit
        """
        )

        st.subheader('')
        st.subheader("Base de données")

        st.markdown(
        """
        Comme énoncé ci-avant, notre client nous a fourni une base de données basée sur la plateforme IMDb. 
        Nous pouvons les retrouver [**ici**](https://datasets.imdbws.com/), l'explicatif des datasets [**là**](https://www.imdb.com/interfaces/).

        Nous laissons à dispositions notre analyse de ces bases de données sur Github dans notre espace collaboratif[**fichier colab**](https://COLLAB)
        """
        )

######################################################################################
######################################################################################
###########################     BERENGER     #########################################
######################################################################################
######################################################################################

    if choice == 'Movie recommandation':
        st.subheader("Movie recommandation")

        # with st.expander("Title"):
        #     mytext = st.text_area("Type Here")
        #     st.write(mytext)
        #     st.success("Hello")

        #st.dataframe(df)
        movies_title_list = df["primaryTitle"].tolist()

        movie_choice = st.selectbox("Movie Title", movies_title_list)
        # with st.expander('Movies DF'):
        #     st.dataframe(df.head(10))

            # Filter
            # img_link = df[df["primaryTitle"] == movie_choice]["img_link"].values[0]
            # title_link = df[df["primaryTitle"] == movie_choice]["primaryTitle"].values
            # genre = df[df["primaryTitle"] == movie_choice]["Comedy"].values
        genre = df[df["primaryTitle"] == movie_choice]["primaryTitle"].tolist()

        #Layout
        # st.write(img_link)
        # st.image(img_link)


        # with c1:
        #     with st.expander("primaryTitle"):
        #         st.write(genre)


        user_choice = genre

        user_choice2 = df[df['primaryTitle'].isin(user_choice)]

        user_choice3 = user_choice2[['Action',
            'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
            'Drama', 'Fantasy', 'History', 'Horror', 'Music', 'Musical', 'Mystery',
            'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'Western']]

        X = df_recommandation[['Action',
            'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
            'Drama', 'Fantasy', 'History', 'Horror', 'Music', 'Musical', 'Mystery',
            'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'Western']]

        distanceKNN = NearestNeighbors(n_neighbors=1).fit(X)

        mewtwo = distanceKNN.kneighbors(user_choice3)

        mewtwo = mewtwo[1].reshape(1,1)[0]
        liste_finale = df_recommandation.iloc[mewtwo]

        for i in range(len(user_choice)):
            liste_base = user_choice[i]
            newlist = liste_finale["primaryTitle"].iloc[i]
            print (f"En remplacement du film {liste_base} je propose {newlist}.")

        st.write(liste_finale[["primaryTitle", "startYear"]])
                


    

    



    elif choice == "Meaningful KPI":
        acteur_par_periode = pd.read_csv("https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Streamlit/acteur_par_periode.csv?token=AU6BUZWYJ6GYLJLQVDQCLZTBSZ2NK")
        link = 'https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Streamlit/top10.csv?token=AU6BUZSEQED65VJVLNSX4FLBS2IYO'
        top10 = pd.read_csv(link)
        presence_acteur = pd.read_csv('https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Streamlit/presence_acteurs.csv?token=AU6BUZRUOZP7577TQEBP5ODBS2IXQ')
        link2 = 'https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Streamlit/film3.csv?token=AU6BUZQSZO7FES64E636CRLBS2IWM'
        film = pd.read_csv(link2)
        link3 = 'https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Streamlit/concat_liste50.csv?token=AU6BUZSY6OPPE25EYFUWFELBS2IS4'
        link4 = 'https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Streamlit/concat_listeTopFilm.csv?token=AU6BUZUX7HJJXUSIP47YANLBS2IVA'
        link5 = 'https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Streamlit/concat_listeTopTV.csv?token=AU6BUZWRESNKYQ36Y652SJLBS2IVW'
        concat_liste_50 = pd.read_csv(link3)
        concat_listeTopFilm = pd.read_csv(link4)
        concat_listeTopTV = pd.read_csv(link5)

        st.title("Projet : recommandations de films")  # add a title

        st.write("Ce projet effectué au sein de l'école Wild Code School a pour but de nous faire créer un moteur de recommandations de films.")

        st.write("Un cinéma en perte de vitesse situé dans la Creuse vous contacte. Il a décidé de passer le cap du digital en créant un site Internet taillé pour les locaux.")

        st.write("Pour commencer, nous devons explorer la base de données afin de répondre aux questions suivantes :")
        st.write("- Quels sont les pays qui produisent le plus de films ?")
        st.write("- Quels sont les acteurs les plus présents ? À quelle période ?")
        st.write("- La durée moyenne des films s’allonge ou se raccourcit avec les années ?")
        st.write("- Les acteurs de série sont-ils les mêmes qu’au cinéma ?")
        st.write("- Les acteurs ont en moyenne quel âge ?")
        st.write("- Quels sont les films les mieux notés ? Partagent-ils des caractéristiques communes ?")


        fig = px.bar(presence_acteur, x="primaryName", y ='index', color = 'index',
            title = 'Quels sont les acteurs les plus présents ?',
            labels = {'primaryName': 'Nombre de films', 'index': 'Acteurs'},
            width=800, height=600)

        fig.update_layout(showlegend=False, title_x=0.5, yaxis={'visible': True}, template='plotly_dark')

        st.plotly_chart(fig)

        fig = px.bar(acteur_par_periode, x = 'count', y="rank", text ='primaryName', color = 'primaryName',
            title = 'Quels sont les acteurs les plus présents par périodes ?',
            labels = {'startYear': 'Période', 'primaryName': 'Acteurs'},
            orientation='h',
            animation_frame="startYear",
            range_x=[0,150],
            range_y=[0,6],
            width=800, height=500)
        
        fig.update_traces(textfont_size=12, textposition='outside')
        fig.update_layout(template='plotly_dark')
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000

        fig.update_layout(showlegend=False, title_x=0.5)

        st.plotly_chart(fig)

        test5 = px.bar(top10, x='Pays', y='Nb de films', color="Nb de films", color_continuous_scale=px.colors.sequential.Viridis, title = 'Pays produisants le plus de film depuis 1960', width=700, height=500, template='plotly_dark')

        st.plotly_chart(test5)



        ######################
        fig = make_subplots(rows=2, cols=2)

        fig.add_trace(go.Line(x = film["startYear"], y=film["runtimeMinutes"]),
                    row=1, col=1)

        fig.add_trace(go.Line(x = film["startYear"], y=film["runtimeMinutes"]),
                    row=1, col=2)

        fig.add_trace(go.Line(x = film["startYear"], y=film["runtimeMinutes"]),
                    row=2, col=1)

        fig.add_trace(go.Line(x = film["startYear"], y=film["runtimeMinutes"]),
                    row=2, col=2)

        fig.update_xaxes(title_text="", row=1, col=1)
        fig.update_yaxes(title_text="", row=1, col=1)

        fig.update_xaxes(title_text="", row=1, col=2)
        fig.update_yaxes(title_text="", row=1, col=2, range=[80, 100])

        fig.update_xaxes(title_text="", row=1, col=1)
        fig.update_yaxes(title_text="", row=2, col=1, range=[50, 100])

        fig.update_xaxes(title_text="", row=1, col=2)
        fig.update_yaxes(title_text="", row=2, col=2, range=[0, 100])

        fig.update_layout(height=1000, width=1400, title_text="Evolution de la durée des films en minutes depuis 1960", title_x=0.5, showlegend=False, template='plotly_dark', autosize=False)

        st.plotly_chart(fig)
        ######################



        fig = px.bar(data_frame = concat_liste_50, x= "primaryName", y="nb", color = 'type', color_discrete_sequence=["darkred", "green"],labels=dict(primaryName="Nom de l'acteur", nb="Nombre de films"))
        fig.update_layout(title_text="Top 20 des acteurs ayant tournés autant au cinéma qu'à la TV", width=1000, height=600, template='plotly_dark')

        st.plotly_chart(fig)



        fig = px.bar(data_frame = concat_listeTopFilm, x= "primaryName", y="nb", color = 'type', color_discrete_sequence=["blue", "lime"], labels=dict(primaryName="Nom de l'acteur", nb="Nombre de films", color = 'type'))
        fig.update_layout(title_text="Top 20 des acteurs ayant tournés le plus du film au cinéma", title_x=0.5, width=1000, height=600, template='plotly_dark')

        st.plotly_chart(fig)


        fig = px.bar(data_frame = concat_listeTopTV, x= "primaryName", y="nb", color = 'type', color_discrete_sequence=["orange", "olive"], labels=dict(primaryName="Nom de l'acteur", nb="Nombre de films"))
        fig.update_layout(title_text="Top 20 des acteurs ayant tournés le plus du film à la télévision", title_x=0.5, width=1000, height=600, template='plotly_dark')

        st.plotly_chart(fig)

        ######################################################################################
        ######################################################################################
        ###########################     AURORE     ###########################################
        ######################################################################################
        ######################################################################################

        country_DF = pd.read_csv('https://raw.githubusercontent.com/ALEMA86/Movie_Recommandation/main/Streamlit/Country.csv')
        col1, col2 = st.columns([1, 2])
        with col1:
            st.title(' ')
            st.markdown(
                """
                Lors de notre analyse de la base de données, nous avons pu observer une grande variété de types d'oeuvres répertoriées par IMDb. 
            
                Notre cliente tenant un cinéma, nous nous sommes attachés à faire un focus sur les films, et avons retenu donc retenu que le type 'movie'.
            

                """
                )

        with col2:
            fig = px.bar(data_frame = country_DF, x= "country", y="nb_mov", labels=dict(country="Pays", nb_mov ="Nombre de films"))
            fig.update_layout(title_text="Palmarès des pays selon la distribution des oeuvres cinématographiques", title_x=0.5, width=1000, height=600, template='plotly_dark')
            st.plotly_chart(fig)





        FULL_DF = pd.read_csv('https://raw.githubusercontent.com/ALEMA86/Movie_Recommandation/main/Streamlit/DF_FULL_KPI.csv')
        col1, col2 = st.columns([1, 1])
        with col1:
            st.title('Note moyenne par genre de films')
            moyenne_genre = pd.pivot_table(FULL_DF,values="averageRating",columns="genre1",aggfunc=np.mean)
 
            # Plot a bar chart using the DF
            ax = moyenne_genre.plot(kind="bar")
            # Get a Matplotlib figure from the axes object for formatting purposes
            fig = ax.get_figure()
            # Change the plot dimensions (width, height)
            fig.set_size_inches(7, 6)
            # Change the axes labels
            ax.set_xlabel("Years")
            ax.set_ylabel("Average Page Views")

            plt.show()

        with col2:
            st.title('Nombre moyen de votes par genre')
            nb_moyen_vote = pd.pivot_table(FULL_DF,values="numVotes",columns="genre1",aggfunc=np.mean)

            # Plot a bar chart using the DF
            ax = nb_moyen_vote.plot(kind="bar")
            # Get a Matplotlib figure from the axes object for formatting purposes
            fig = ax.get_figure()
            # Change the plot dimensions (width, height)
            fig.set_size_inches(7, 6)
            # Change the axes labels
            ax.set_xlabel("Years")
            ax.set_ylabel("Average Page Views")

            plt.show()


        fig = px.bar(FULL_DF, x="numVotes", y ='genre1', color = 'index',aggfunc=np.mean,
        title = 'Quels sont les acteurs les plus présents ?',labels = {'primaryName': 'Nombre de films', 'index': 'Acteurs'}, width=800, height=600)
        fig.update_layout(showlegend=False, title_x=0.5, yaxis={'visible': True}, template='plotly_dark')
        st.plotly_chart(fig)


main()



