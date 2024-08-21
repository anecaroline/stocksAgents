#!/usr/bin/env python
# coding: utf-8

# In[28]:
#python -m jupyter nbconvert --to script crewai-stocks.ipynb -> criar um arquivo.py
#jupyter nbconvert --to script crewai-stocks.ipynb


# pip install langchain_openai
# pip install langchain_community

# yfinance==0.2.41
# crewai==0.28.8
# langchain==0.1.20
# langchain-openai==0.1.7
# langchain-community==0.0.38
# duckduckgo-search==5.3.0


# In[29]:


import json
import os
from datetime import datetime

import yfinance as yf

from crewai import Agent,Task,Crew,Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI

from langchain_community.tools import DuckDuckGoSearchResults

#from IPython.display import Markdown
import streamlit as st



# In[31]:


#criando Yahool finance Tool
def fetch_stock_price(ticket):
    stock=yf.download(ticket,start="2023-08-08",end="2024-08-08")
    return stock

yahool_finance_tool=Tool(
    name = "Yahoo Finance Tool",
    description = "Fetches stocks prices for {ticket} from the last year about a specific company from Yahoo Finance API",
    func=lambda ticket: fetch_stock_price(ticket)
)


# In[32]:


# response = yahool_finance_tool.run("AAPL")


# In[33]:


# print(response)


# In[34]:


#importando OPENAI LLM - GPT
#criar conta para poder usar
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")


# In[35]:


stockPriceAnalyst= Agent(
    role="Senior stock price Analyst",
    goal="Find the {ticket} stock price and analyses trends",
    backstory="""You're a highly experienced in analyzing the price of an specific stock
    and make predictions about its future price""",
    verbose=True,
    llm = llm,
    max_iter=5,
    memory=True,
    tools=[yahool_finance_tool],
    allow_delegation = False,
)


# In[37]:


getStockPrice = Task(
    description= "Analyze the stock {ticket} price history and create a trend analyses of up, down or sideways",
    expected_output = """" Specify the current trend stock price - up, down or sideways. 
    eg. stock= 'APPL, price UP'
""",
    agent= stockPriceAnalyst
)


# In[38]:


#importando a tool de search
search_tool = DuckDuckGoSearchResults(backend='news',num_results=10)


# In[39]:


newsAnalyst = Agent(
    role="Stock News Analyst",
    goal="""Create a short summary of the market newa to the stock {ticket} company.Specify the current trend-up,down or sideways withs the news context.
    For each request stock asset,specify a number between 0 and 100,where 0 is extreme fear and 100 is extreme greed.""",
    backstory="""You're experienced in analyzing the market trends and news and have traked assest for more then 10 years.

    You're also master leves analyts in the tradicional markets and have deep understanding of human psychology.

    you understand news,their titles and information,but you look at those with a health dose of skepticism.
    """,
    verbose=True,
    llm = llm,
    max_iter=5,
    memory=True,
    tools=[search_tool],
    allow_delegation = True,
)


# In[41]:


get_news = Task(
    description= f"""Take the stock and always include BTC to it (if not request).
    Use the search tool to search each one individually. 

    The current date is {datetime.now()}.

    Compose the results into a helpfull report""",
    expected_output = """"A summary of the overall market and one sentence summary for each request asset. 
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
""",
    agent= newsAnalyst
)


# In[45]:


stockAnalystWrite = Agent(
    role = "Senior Stock Analyts Writer",
    goal= """"Analyze the trends price and news and write an insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend. """,
    backstory= """You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories
    and narratives that resonate with wider audiences. 

    You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses. 
    You're able to hold multiple opinions when analyzing anything.
""",
    verbose = True,
    llm=llm,
    max_iter = 5,
    memory=True,
    allow_delegation = True
)


# In[46]:


writeAnalyses = Task(
    description = """Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} company
    that is brief and highlights the most important points.
    Focus on the stock price trend, news and fear/great score.What are the near future considerations?
    Include the previous analyses pf stock trend and news summary""",
    expected_output="""An eloquent 3 paragraphs nwesletter formated as markdow in an easy readable manner.It should contain:
    - 3 bullets executive summary
    - introduction -set the overall picture and spike up the interest
    - man part provides the meat of the analysis includinf the news summary and fead/great scores
    - summary - key facts and concrete future trend prediction - up,down or sideways.

    """,
    agent = stockAnalystWrite,
    context = [getStockPrice,get_news]

)


# In[48]:


crew = Crew(
    agents = [stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
    tasks = [getStockPrice, get_news, writeAnalyses],
    verbose = True,
    process= Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)


# In[49]:


#results = crew.kickoff(inputs={'ticket':'AAPL'})


# In[54]:


# print(type(results))


# In[56]:


# Accessing the crew output
# print(f"Raw Output: {results.raw}")
# if results.json_dict:
#     print(f"JSON Output: {json.dumps(results.json_dict, indent=2)}")
# if results.pydantic:
#     print(f"Pydantic Output: {results.pydantic}")
# print(f"Tasks Output: {results.tasks_output}")
# print(f"Token Usage: {results.token_usage}")


# In[60]:


# list(results.keys())


# In[61]:


#results['final_outputs']


# In[62]:


# len(results['tasks_outputs'])


# In[63]:


# Markdown(results['final_output'])

with st.sidebar:
    st.header('Enter the Stock to Research')

    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label = "Run Research")
if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results= crew.kickoff(inputs={'ticket': topic})

        st.subheader("Results of research:")
        st.write(results['final_output'])