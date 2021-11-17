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
import matplotlib.pyplot as plt


######################################################################################
######################################################################################
###########################     DONNEES    ###########################################
######################################################################################
######################################################################################

#df_recommandation = pd.read_csv('https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Database_projet/df_recommendation.csv?token=AU6BUZUA5UESEPKRRJQIESLBS53UU')
df = pd.read_csv('https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Database_projet/df_base.csv?token=AU6BUZWHN456IAMFBUWFFSDBTELCU')
FULL_DF = pd.read_csv('https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Aurore/KPI/DF_FULL_GENRES211117.csv?token=AUTGRH2WPDMVJ7DBATL3KWDBTYLIM')

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
    menu = ["Présentation du Projet", "Movie recommandation", "Analyses et KPI"]

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

        Tous les quatre formons l'équipe ABC'S Data.
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

        Elle fait appel à nous car elle est désespérée. Son cinéma ne fait pas de bénéfice, ses créanciers sont à sa porte et ses problèmes financiers sont tels qu'elle a dû demandé un nouveau prêt dans une banque alors que c'est contre ses principes.

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
        Nous devons fournir à notre client les outils d’analyse de la base de données issue de **IMDB**.
       
        Il nous est demandé de :
        """
        )
        st.markdown(
        """ 
        - Faire une rapide présentation de la base de données (sur notre espace collaboratif sur Github)
        """
        )
        st.markdown(
        """ 
        - Fournir à notre client quelques statistiques sur les films :
        """
        )
        st.markdown(
        """ 
            *Films : types, durées...
        """
        )
        st.markdown(
        """ 
            *Acteurs : nombre de films, type de films...
        """
        )
        st.markdown(
        """ 
        - Présenter les TOP 10 des films par années et genre
        """
        )
        st.markdown(
        """ 
        - Présenter les TOP 5 des acteurs/actrices par années et genre
        """
        )
        st.markdown(
        """ 
        - Retourner une liste de films recommandés en fonction d'IDs ou de noms de films choisis par un utilisateur
        """
        )
        st.markdown(
        """ 
        - Il faudra entraîner des outils de Machine Learning : 
        """
        )
        st.markdown(
        """ 
	        *Recommandation de films proches d’un film cible grâce à un modèle de **KNN**
        """
        )
        st.markdown(
        """ 
	        *Proposition d’une rétrospective avec un modèle de **Régression Logistique**
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
                
######################################################################################
######################################################################################
###########################     AURORE     ###########################################
######################################################################################
######################################################################################   



    elif choice == "Analyses et KPI":

        
        
        
        
        
        link2 = 'https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Streamlit/film3.csv?token=AU6BUZQSZO7FES64E636CRLBS2IWM'
        film = pd.read_csv(link2)
        link3 = 'https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Streamlit/concat_liste50.csv?token=AU6BUZSY6OPPE25EYFUWFELBS2IS4'
        link4 = 'https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Streamlit/concat_listeTopFilm.csv?token=AU6BUZUX7HJJXUSIP47YANLBS2IVA'
        link5 = 'https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Streamlit/concat_listeTopTV.csv?token=AU6BUZWRESNKYQ36Y652SJLBS2IVW'
        concat_liste_50 = pd.read_csv(link3)
        concat_listeTopFilm = pd.read_csv(link4)
        concat_listeTopTV = pd.read_csv(link5)


        #######################################
        ########  Introduction     ############
        #######################################

        st.title("Projet ABC'S : Recommandations de Films")  # add a title
        st.write("")
        st.subheader("Analyses de la base de données et KPI") # add a subtitle

        st.write("Comme énoncé dans notre partie **'Présentation du Projet'**, il nous est demandé de :")
        st.markdown(
        """
        - Faire une rapide présentation de la base de données (que vous pouvez retrouver [ici](https://github.com/BerengerQueune/ABC-Data/blob/main/Aurore/Analyses_BDD_Etape%201.ipynb))
        - Faire une analyse complète de la base de données, en répondant aux questions suivantes :
            * Quels sont les pays qui distribuent le plus de films ?
            * Quels sont les acteurs les plus présents ? À quelle période ?
            * La durée moyenne des films s’allonge ou se raccourcit avec les années ?
            * Les acteurs de série sont-ils les mêmes qu’au cinéma ? 
            * Les acteurs ont en moyenne quel âge ? 
            * Quels sont les films les mieux notés ? Partagent-ils des caractéristiques communes ?
        """
        )
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        #######################################
        ########    GRAPHIQUES     ############
        #######################################


        ########  Q01 -Christophe  ############
        st.subheader("Quels sont les pays qui distribuent le plus de films ?") # add a subtitle
        st.image("https://i.ibb.co/bHkZJb7/B-mod.png") 

        top10 = pd.read_csv('https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Streamlit/top10.csv?token=AU6BUZSEQED65VJVLNSX4FLBS2IYO')

        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(' ')
            st.markdown(
                """
                Le dataset a été élaboré à partir de deux fichiers : title.basics et title.akas.

                Lors de notre analyse de la base de données, nous avons pu observer une grande variété de types d'oeuvres répertoriées par IMDb. 

                Ainsi, à partir de title.basics, il a été choisi de ne retenir que les films ('movie') et téléfilms ('tvMovie) réalisés après 1960, limitant notre périmètre d’analyse aux films les plus récents. Les courts-métrages (“short”) ont également été retirés.
                Les lignes n'ayant pas de données pour les items suivants ont été supprimées de notre DataFrame: année de réalisation ('startYear'), de durée ('runtimeMinutes') ou de genres ('genres').

                De la même façon, les films qui n’ont pas de région dans le fichiers title.akas ont été supprimés.

                Une jointure a été réalisée entre les deux DataFrame afin d’ajouter la région aux colonnes de la base de données title.basics.

                Afin de réaliser le graphique, un [dataframe attitré]('https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Streamlit/top10.csv?token=AU6BUZSEQED65VJVLNSX4FLBS2IYO') reprenant  le top 10 des pays ayant distribué le plus de films et téléfilms a été produit.

                [Lien Notebook](https://github.com/BerengerQueune/ABC-Data/blob/main/Christophe/Projet%202%20-%20Quels%20sont%20les%20pays%20qui%20produisent%20le%20plus%20de%20films.ipynb)

                """
                )

        with col2:
            top10Graph = px.bar(top10, x='Pays', y='Nb de films', color="Nb de films")
            top10Graph.update_layout(title_text="Palmarès des pays selon la distribution des oeuvres cinématographiques", title_x=0.5, width=1000, height=600, template='plotly_dark')
            st.plotly_chart(top10Graph)

        st.write("")
        st.markdown("""
                Ce graphique montre clairement une prédominance des USA dans le nombre de films distribués, puisque leur nombre dépasse la somme de ceux réalisés dans les deux pays suivants à savoir la Grande-Bretagne et la France.               
                A noter également que l’on retrouve en troisième position des films dont l’origine est inconnue (XWW = World Wide).
                On note également que trois des 5 continents sont représentés dans le top10.
                La France confirme cependant sa position de cinéphile en étant dans le top 3 si nous excluons la région 'XWW'.
                """
                )
        st.write(' ')
        st.write(' ')
        st.write(' ')
        ########  Q02 -Bérenger  ############
        st.subheader("Quels sont les acteurs les plus présents ?") # add a subtitle
 
        presence_acteur = pd.read_csv('https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Database_projet/presence_acteurs.csv?token=AU6BUZU76KCNKK6X5NKIZ6DBTZPVI')

        col1, col2 = st.columns([2, 1])
        with col2:
            st.write(' ')
            st.markdown(
                """
                Le dataset a été élaboré à partir de 3 fichiers : name.basics.tsv, title.principals.tsv et title.basics.tsv.

                Nous avons nettoyé la base de données de la façon suivante :
                - dans le df relatif à 'title.principals.tsv', nous avons gardé les colonnes 'tconst', 'titleType', 'startYear', 'runtimeMinutes' et 'genres'
                    - dans la colonne 'category' nous avons gardé les 'actor' et 'actress'
                    - dans la colonne 'character', nous avons supprimé les 'backslash N', les 'Narrator', 'Various' et 'Additional Voices'
                - dans le df relatif à 'title.basics.tsv', nous avons gardé les colonnes 'tconst', 'nconst', 'category' et 'characters'
                **EN ATTENTE EN ATTENTE EN ATTENTE EN ATTENTE EN ATTENTE EN ATTENTE**

                Afin de réaliser le graphique, un [dataframe attitré]('https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Database_projet/presence_acteurs.csv?token=AU6BUZU76KCNKK6X5NKIZ6DBTZPVI') reprenant les 20 acteurs les plus présents quelle que soit l'époque a été produit.

                [Lien Notebook](https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Quels_sont_les_acteurs_les_plus_pr%C3%A9sents.ipynb?token=AUTGRHYRYCTRXKOZDDLVIELBTZL3I)

                """
                )

        with col1:
            fig = px.bar(presence_acteur, x="primaryName", y ='index', color = 'index',
            title = 'Quels sont les acteurs les plus présents ?',
            labels = {'primaryName': 'Nombre de films', 'index': 'Acteurs'},
            width=800, height=600)

            fig.update_layout(showlegend=False, title_x=0.5, yaxis={'visible': True}, template='plotly_dark')

            st.plotly_chart(fig)

        st.write("")
        st.markdown("""
                **EN ATTENTE EN ATTENTE EN ATTENTE EN ATTENTE EN ATTENTE EN ATTENTE**
                """
                )


        st.write(' ')
        st.write(' ')
        st.write(' ')
        ########  Q03 -Bérenger  ############
        st.subheader("Quels sont les acteurs les plus présents, à quelle période ?") # add a subtitle
        acteur_par_periode = pd.read_csv("https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Streamlit/acteur_par_periode.csv?token=AU6BUZWYJ6GYLJLQVDQCLZTBSZ2NK")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(' ')
            st.markdown(
                """
                
                **EN ATTENTE EN ATTENTE EN ATTENTE EN ATTENTE EN ATTENTE EN ATTENTE**

                Afin de réaliser le graphique, un [dataframe attitré]('https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Streamlit/acteur_par_periode.csv?token=AU6BUZWYJ6GYLJLQVDQCLZTBSZ2NK') reprenant les 5 acteurs les plus présents pour chaque décennies depuis 1910.

                [Lien Notebook](https://raw.githubusercontent.com/BerengerQueune/ABC-Data/main/Berenger/Quels_sont_les_acteurs_les_plus_pr%C3%A9sents.ipynb?token=AUTGRHYRYCTRXKOZDDLVIELBTZL3I)

                """
                )

        with col2:
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

        st.write("")
        st.markdown("""
                **EN ATTENTE EN ATTENTE EN ATTENTE EN ATTENTE EN ATTENTE EN ATTENTE**
                """
                )











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




        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.title('Note moyenne par genre de films')
            moyenne_genre = pd.pivot_table(FULL_DF,values="averageRating",columns="genre1",aggfunc=np.mean)
            moyenne_genre_unstacked = moyenne_genre.unstack().unstack()
            moyenne_genre_unstacked =moyenne_genre_unstacked.sort_values('averageRating')

            Genres = moyenne_genre_unstacked.index
            moyenne = moyenne_genre_unstacked['averageRating']

            fig = px.bar(moyenne_genre_unstacked, x=Genres, y =moyenne, labels = {'averageRating': 'Note moyenne', 'genre1': 'Genres de 1er rang'},color = moyenne_genre_unstacked.index,title = 'Note moyenne par genre de films ',width=600, height=450)
            fig.update_layout(showlegend=False, title_x=0.5, yaxis={'visible': True}, template='plotly_dark')
            st.plotly_chart(fig)

        with col2:
            st.title('Nombre moyen de votes par genre')
            nb_moyen_votes = pd.pivot_table(FULL_DF,values="numVotes",columns="genre1",aggfunc=np.mean)
            nb_moyen_votes_unstacked = nb_moyen_votes.unstack().unstack()
            nb_moyen_votes_unstacked = nb_moyen_votes_unstacked.sort_values('numVotes').round()

            genres = nb_moyen_votes_unstacked.index
            nb_votes = nb_moyen_votes_unstacked['numVotes']

            fig = px.bar(nb_moyen_votes_unstacked, x=genres, y =nb_votes, labels = {'numVotes': 'Nombre moyen de votes', 'genre1': 'Genres de 1er rang'}, color = nb_moyen_votes_unstacked.index,title = "Nombre moyen de votes par genre",width=600, height=450)
            fig.update_layout(showlegend=False, title_x=0.5, yaxis={'visible': True}, template='plotly_dark')
            
            st.plotly_chart(fig)

    
        






main()



