from flask import Flask, render_template, request, jsonify
from datetime import datetime
import pandas as pd
import os
from typing import Dict
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from typing import List, Dict
import pandas as pd
import requests
from datetime import datetime, timedelta
# Weather data storage functions
def save_weather_data(df: pd.DataFrame, filename: str = 'weather_data.csv'):
    """Save weather data to CSV file"""
    df.to_csv(filename, index=False)
    print(f"Weather data saved to {filename}")
def load_weather_data(filename: str = 'weather_data.csv') -> pd.DataFrame:
    """Load weather data from CSV file"""
    if os.path.exists(filename):
        return pd.read_csv(filename)
    return None

# Create Flask application
app = Flask(__name__)
class WeatherRAG:
    def __init__(self, api_key: str):
        """
        Initialize the Weather RAG system
        """
        self.api_key = api_key
        self.llm = self._setup_llm()
        self.retriever_prompt = self._create_retriever_prompt()
        self.qa_prompt = self._create_qa_prompt()
        self.parser = self._get_output_parser()
        
    def _setup_llm(self):
        """Initialize Gemini Pro LLM"""
        return GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.api_key,
            temperature=0.1
        )
    
    def _create_retriever_prompt(self):
        """Create prompt template for the retriever"""
        template = """
        You are a weather data assistant. Based on the following date/time query, 
        help me find relevant weather information from a dataset.

        Query: {query}

        The data has the following format:
        - datetime: start time of the weather record
        - to: end time of the weather record
        - city: city name
        - country: country code
        - temperature: temperature in Celsius
        - feels_like: perceived temperature
        - humidity: humidity percentage
        - weather_desc: weather description
        - wind_speed: wind speed
        - wind_direction: wind direction in degrees
        - precipitation: precipitation amount
        - pressure: atmospheric pressure
        - clouds: cloud coverage percentage

        Extract the following information from the query:
        1. Start datetime (if mentioned)
        2. End datetime (if mentioned)
        3. Any specific weather attributes requested

        {format_instructions}
        """
        return PromptTemplate(
            input_variables=["query"],
            partial_variables={"format_instructions": self._get_output_parser().get_format_instructions()},
            template=template
        )
    
    def _create_qa_prompt(self):
        """Create prompt template for question answering"""
        template = """
        You are a helpful weather assistant. Using the provided weather data, answer the user's question.
        
        Weather Data:
        {context}
        
        User Question: {question}
        
        Please provide a clear, informative answer that:
        1. Directly addresses the user's question
        2. Includes specific numbers and measurements when relevant
        3. Provides context about the weather conditions
        4. Highlights any notable patterns or changes
        
        Answer:
        """
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    def _get_output_parser(self):
        """Create structured output parser"""
        response_schemas = [
            ResponseSchema(name="start_datetime", 
                          description="The start datetime mentioned in the query, in YYYY-MM-DD HH:mm:ss format"),
            ResponseSchema(name="end_datetime", 
                          description="The end datetime mentioned in the query, in YYYY-MM-DD HH:mm:ss format"),
            ResponseSchema(name="attributes", 
                          description="List of specific weather attributes requested")
        ]
        return StructuredOutputParser.from_response_schemas(response_schemas)

    def fetch_weather_data(self, city: str) -> Dict:
        """Fetch weather forecast data from OpenWeatherMap API"""
        url = f"http://api.openweathermap.org/data/2.5/forecast"
        params = {
            'q': city,
            'appid': 'THE_API_KEY_OF_OPENWEATHERMAP',  # Consider making this configurable
            'units': 'metric'
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch weather data: {response.status_code}")
        return response.json()

    def process_weather_data(self, raw_data: Dict) -> pd.DataFrame:
        """Process raw weather data into a structured format"""
        processed_data = []
        city_name = raw_data['city']['name']
        country = raw_data['city']['country']
        
        for item in raw_data['list']:
            processed_item = {
                'datetime': datetime.fromtimestamp(item['dt']),
                'to': datetime.fromtimestamp(item['dt']) + timedelta(hours=3),
                'city': city_name,
                'country': country,
                'temperature': item['main']['temp'],
                'feels_like': item['main']['feels_like'],
                'humidity': item['main']['humidity'],
                'weather_desc': item['weather'][0]['description'],
                'wind_speed': item['wind']['speed'],
                'wind_direction': item['wind']['deg'],
                'precipitation': item.get('rain', {}).get('3h', 0),
                'pressure': item['main']['pressure'],
                'clouds': item['clouds']['all']
            }
            processed_data.append(processed_item)
        return pd.DataFrame(processed_data)

    def retrieve_relevant_data(self, query: str, df: pd.DataFrame):
        """Retrieve relevant weather data based on the query"""
        _input = self.retriever_prompt.format_prompt(query=query)
        response = self.llm.invoke(_input.to_string())
        
        try:
            parsed_output = self.parser.parse(response)
            
            # Filter dataframe based on datetime range
            mask = pd.Series(True, index=df.index)
            
            if parsed_output.get('start_datetime'):
                start_dt = pd.to_datetime(parsed_output['start_datetime'])
                mask &= (df['datetime'] >= start_dt)
                
            if parsed_output.get('end_datetime'):
                end_dt = pd.to_datetime(parsed_output['end_datetime'])
                mask &= (df['to'] <= end_dt)
                
            filtered_df = df[mask]
            
            return filtered_df
            
        except Exception as e:
            print(f"Retrieval error: {str(e)}")
            print(f"Raw LLM response: {response}")
            return df  # Return full dataset if parsing fails
    
    def generate_answer(self, question: str, context_df: pd.DataFrame):
        """Generate an answer based on the question and retrieved context"""
        context = context_df.to_string()
        _input = self.qa_prompt.format_prompt(
            context=context,
            question=question
        )
        
        try:
            response = self.llm.invoke(_input.to_string())
            return response
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def query(self, user_question: str, city: str):
        """
        Main query method that orchestrates the entire RAG process
        """
        try:
            # 1. Fetch fresh weather data
            raw_data = self.fetch_weather_data(city)
            df = self.process_weather_data(raw_data)
            
            # 2. Retrieve relevant data
            relevant_data = self.retrieve_relevant_data(user_question, df)
            
            # 3. Generate answer
            answer = self.generate_answer(user_question, relevant_data)
            
            return answer
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
# Initialize RAG system
weather_rag = None  # We'll initialize this with your WeatherRAG class

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    
    try:
        # Load the stored weather data
        df = load_weather_data()
        if df is None:
            return jsonify({'error': 'No weather data available'})
        
        # Generate response using your RAG system
        response = weather_rag.generate_answer(user_message, df)
        
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Initialize the RAG system with your API key
    api_key = "YOUR_GEMINI_API_KEY"
    weather_rag = WeatherRAG(api_key)
    
    # Fetch and save initial weather data
    raw_data = weather_rag.fetch_weather_data("Alger")
    df = weather_rag.process_weather_data(raw_data)
    save_weather_data(df)
    
    app.run(debug=True)