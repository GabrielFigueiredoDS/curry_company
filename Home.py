import streamlit as st
from PIL import Image

# ==================================
# Configurações Página 
# ==================================
st.set_page_config(page_title='Home', page_icon="📈", layout='wide')

image = Image.open('logo.png')
st.sidebar.image(image, width=220)

st.sidebar.markdown('# Cury Company')
st.sidebar.markdown('## Fastest Delivery in Town')
st.sidebar.markdown("""---""")

st.write("# Curry Company Groeth Dashboard")

st.markdown(
    """
    Groeth Dashboard foi construído para acompanhar as métricas de crescimento dos Entregadores e Restaurantes.
    ### Como utilizar esse Groth Dashboard?
    - Visão Empresa:
        - Visão Gerencial: Métricas gerais de comportamento.
        - Visão Tática: Indicadores semanais de crescimento.
        - Visão Geográfica: Insights de geolocalização.
    - Visão Entregador: 
        - Acompanhamento dos indicadores semanais de crescimento.
    - Visão Restaurante:
        - Indicaroes semanais de crescimento dos restaurantes.
    ### Ask for Help
    - Time de Data Science no Discord
    
            @Gabriel Figueiredo
""")
