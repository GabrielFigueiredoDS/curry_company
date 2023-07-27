# ==================================
# Liberies 
# ==================================

from haversine import haversine
import plotly.express as px
import pandas as pd
import re
import streamlit as st
from PIL import Image
import folium
from streamlit_folium import folium_static

# ==================================
# Funções 
# =================================

def clean_code(df):
    """Esta função tem a responsabilidade de limpar o data frame
    
    Tipos de limpeza:
    1. Remoção dos dados NaN
    2. Mudançã do tipo da coluna de dados
    3. Remoção dos espaços das variáveis de texto
    4. Formatação da coluna de datas
    5. Limpeza da coluna tempo (remoção da variável numérica)
    
    Input: DataFrame
    Output: DataFrame 
    """
    # 1. convertando a coluna Age de testo para numero
    linhas_vazias = df['Delivery_person_Age'] != 'NaN '
    df = df.loc[linhas_vazias, :].copy()

    linhas_vazias = df['Road_traffic_density'] != 'NaN '
    df = df.loc[linhas_vazias, :].copy()

    linhas_vazias = df['City'] != 'NaN '
    df = df.loc[linhas_vazias, :].copy()

    linhas_vazias = df['Festival'] != 'NaN '
    df = df.loc[linhas_vazias, :].copy()

    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype( int )

    # 2. convertendo a coluna Tatings de testo para numero decimal (float)
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype( float )

    # 3. convertendo a coluna order_date de testo para data
    df['Order_Date'] = pd.to_datetime( df['Order_Date'], format='%d-%m-%Y' )

    # 4. convertendo multiple_deliveries de testo para numero inteiro (int)
    linhas_vazias = df['multiple_deliveries'] != 'NaN '
    df = df.loc[linhas_vazias, :].copy()
    df['multiple_deliveries'] = df['multiple_deliveries'].astype( int )

    # 6. Removendo os espaços dentro de strings/testos/object
    df.loc[:, 'ID'] = df.loc[:, 'ID'].str.strip()
    df.loc[:, 'Road_traffic_density'] = df.loc[:, 'Road_traffic_density'].str.strip()
    df.loc[:, 'Type_of_order'] = df.loc[:, 'Type_of_order'].str.strip()
    df.loc[:, 'Type_of_vehicle'] = df.loc[:, 'Type_of_vehicle'].str.strip()
    df.loc[:, 'City'] = df.loc[:, 'City'].str.strip()
    df.loc[:, 'Festival'] = df.loc[:, 'Festival'].str.strip()

    # 7. Retirando os numeros da coluna Time_taken(min)
    df['Time_taken(min)'] = df['Time_taken(min)'].apply( lambda x: x.split( '(min) ')[1])
    df['Time_taken(min)'] = df['Time_taken(min)'].astype(int)

    # Remove os NA que forem np.na
    df = df.dropna()
    
    return df

def top_delivers(df, top_asc):
    """ Esta função tem a responsabilidade DataFrame, agrupado por
    tipo de cidade e ordenado pelo tempo de entregar dos entregadores.
    
    Paramentros:
    1. top_asc = True 
        Retorna os 10 entregadores mais rapidos.
    2. top_asc = False 
        Retorna os 10 entregadores mais lentos
        .
    Input: DataFrame
    Output: DataFrame
    """
    # Os 10 entregadores mais lentos por tipo de tráfico 
    df_grouped = (df.loc[:, ['Delivery_person_ID', 'City', 'Time_taken(min)']].groupby(['City', 'Delivery_person_ID'])
                  .mean()
                  .sort_values(['City', 'Time_taken(min)'], ascending=top_asc)
                  .reset_index())

    df_grouped1 = df_grouped.loc[df_grouped['City'] == 'Metropolitian', :].head(10)
    df_grouped2 = df_grouped.loc[df_grouped['City'] == 'Urban', :].head(10)
    df_grouped3 = df_grouped.loc[df_grouped['City'] == 'Semi-Urban', :].head(10)

    df_grouped4 = pd.concat( [df_grouped1, df_grouped2, df_grouped3] ).reset_index(drop=True)

    return df_grouped4

def delivery_mean_std_cols(df, coluna):
    """ Esta função tem a responsabilidade de criar um DataFrame, selecionando a
    coluna ['Delivery_person_Ratings'] pela coluna escolhida pelo usuário. em seguida
    é feito um agrupamento e pela coluna escolhida epelo usuário e calculado a média
    e o desvio padrão.

    Paramentros:
    coluna = Usuário deve escolher a coluna que deseja agrupar com a coluna
    ['Delivery_person_Ratings'] para ser realizado a média e desvio padrão. 
    Obs: A coluna escolhida logicamente deve ser do tipo numerico. 

    Input: DataFrame
    Output: DataFrame
    """

    df_grouped = ( df.loc[:, ['Delivery_person_Ratings', coluna]]
                                     .groupby(coluna)
                                     .agg({'Delivery_person_Ratings':['mean', 'std']} ))

    # rename columns
    df_grouped.columns = ['delivery_mean', 'delivery_std']

    # reset index
    df_grouped = df_grouped.reset_index()

    return df_grouped

def retings_delivery(df):
    """ Esta função tem a responsabilidade de criar um DataFrame, selecionando a
    coluna ['Delivery_person_Ratings'] pela coluna ['Delivery_person_ID']. Em seguida
    é feito um agrupamento por ['Delivery_person_ID'] e calculado a média.

    Input: DataFrame
    Output: DataFrame
    """
    df_grouped = (df.loc[:, ['Delivery_person_Ratings', 'Delivery_person_ID']]
                                     .groupby(['Delivery_person_ID'])
                                     .mean()
                                     .reset_index())
    return df_grouped
# ----------------------- Inicio da Estrutura lógica do código-------------------#

# ==================================
# Configurações Página 
# ==================================
st.set_page_config(page_title='Visão Entregadores', page_icon="🚚", layout='wide')

# ==================================
# DataSet 
# ==================================
df_raw = pd.read_csv('dataset/train.csv')

# Fazendo uma cópia do DataFrame Lido
df = df_raw.copy()

# ==================================
# Limpeza dos dados 
# ==================================
df = clean_code( df )

# ==================================
# Barra Lateral
# ==================================

st.header('Marketplace - Visão Entregadores')

image_path = 'logo.png'
image = Image.open(image_path)
st.sidebar.image(image, width=220)

st.sidebar.markdown('# Cury Company')
st.sidebar.markdown('## Fastest Delivery in Town')
st.sidebar.markdown("""---""")

st.sidebar.markdown('## Selecione uma data limite')

date_slider = st.sidebar.slider(
    'Até qual valor?',
    value=pd.to_datetime(2022,4,13),
    min_value=pd.to_datetime( 2022, 2, 11),
    max_value=pd.to_datetime( 2022, 4, 6),
    format='DD-MM-YYYY')

st.header( date_slider )
st.sidebar.markdown("""---""")

traffic_options = st.sidebar.multiselect(
    'Quais as condições do trânsito',
    ['Low', 'Medium', 'High', 'Jam'],
    default=['Low', 'Medium', 'High', 'Jam'])


st.sidebar.markdown("""---""")
st.sidebar.markdown('### Powered by Gabriel Figueirêdo')

# Filtro data
linhas_selecionadas = df['Order_Date'] < date_slider
df = df.loc[linhas_selecionadas, :]

# Filtro transito
linhas_selecionadas = df['Road_traffic_density'].isin(traffic_options)
df = df.loc[linhas_selecionadas, :]

# ==================================
# Layout no Streamlit
# ==================================
tab1, tab2, tab3 = st.tabs(['Visão Gerencial', '_', '_'])

with tab1:
    with st.container():
        st.title('Overall Metrics')
        
        col1, col2, col3, col4 = st.columns(4, gap='large')
        with col1:
            # A maior idade dos entregadores
            maior_idade = df.loc[:, "Delivery_person_Age"].max()
            col1.metric('Maior de idade', maior_idade)
        
        with col2:
            # A menor idade dos entregadores
            menor_idade = df.loc[:, "Delivery_person_Age"].min()
            col2.metric('Menor idade', menor_idade)
        
        with col3:
            # A melhor condição de veículos
            melhor_condicao = df.loc[:, "Vehicle_condition"].max()
            col3.metric('Melhor condição', melhor_condicao)
        
        with col4:
            # A pior condição de véiuclos
            pior_condicao = df.loc[:, "Vehicle_condition"].min()
            col4.metric('Pior condição', pior_condicao)
    
    with st.container():
        st.markdown("""---""")
        st.title('Avaliações')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('#### Avaliaçaõ medias por Entregador')
            df_avg_ratings_per_deliverery = retings_delivery(df)
            st.dataframe(df_avg_ratings_per_deliverery, height=505)

        with col2:
            st.markdown('#### Avaliação media por transito')
            df_avg_std_rating_by_traffic = delivery_mean_std_cols(df,'Road_traffic_density')
            st.dataframe(df_avg_std_rating_by_traffic)
            
            st.markdown('#### Avaliação media por clima')
            df_avg_rating_by_weather = delivery_mean_std_cols(df, 'Weatherconditions')
            st.dataframe(df_avg_rating_by_weather)
            
            
        with st.container():
            st.markdown("""---""")
            st.title('Velocidade de Entrega')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('#### Top Entregadores mais rapidos')
                df_grouped = top_delivers(df, top_asc=True)
                st.dataframe(df_grouped)
            
            with col2:
                st.markdown('#### Top Entregadores mais lentos')
                df_grouped = top_delivers(df, top_asc=False)
                st.dataframe(df_grouped)
                
            
