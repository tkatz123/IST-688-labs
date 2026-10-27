import streamlit as st
import requests
import json
import os
from openai import OpenAI, AuthenticationError
from langchain.agents import create_agent

def get_local_time(location):
    '''
    Look up the current local clock time and timezone for a location using wttr.in.

    Args:
        location (str): A city name or place query understood by wttr.in (e.g. 'Syracuse, NY').

    Returns:
        tuple: A 3-tuple of (local_time, timezone, hour):
            - local_time (str): The local time formatted as 12-hour time, e.g. '3:57 PM'.
            - timezone (str): The timezone abbreviation, e.g. 'EDT'.
            - hour (int): The local hour in 24-hour format (0-23).
        Returns (None, None, None) if the request fails or the response can't be parsed.
    '''
    # The j1 feed's observation_time is UTC, which reads as a future time in the Americas.
    # This one-line endpoint returns the city's actual local clock time and timezone.
    try:
        response = requests.get(f'https://wttr.in/{location}?format=%T|%Z', timeout = 10)
        if response.status_code != 200:
            return None, None, None
        # raw_time looks like '15:57:46-0400'
        raw_time, timezone = response.text.strip().split('|')
        hour, minute = int(raw_time[:2]), raw_time[3:5]
        suffix = 'AM' if hour < 12 else 'PM'
        return f'{(hour % 12) or 12}:{minute} {suffix}', timezone, hour
    except (requests.RequestException, ValueError, IndexError):
        return None, None, None

def get_current_weather(location):
    '''
    Fetch current conditions, today's hourly forecast, and the upcoming daily forecast
    for a location from wttr.in. All temperatures are in Fahrenheit and speeds in mph.

    Args:
        location (str): A city name or place query understood by wttr.in (e.g. 'Syracuse, NY').

    Returns:
        dict: Weather data for the location, containing:
            - Location info: 'location', 'city', 'region', 'country'.
            - Current conditions: 'local_time', 'timezone', 'observation_time_utc',
              'temperature', 'feels_like', 'description', 'humidity', 'cloud_cover',
              'wind_speed', 'wind_direction', 'uv_index', 'precipitation_inches',
              'visibility_miles', 'pressure'.
            - Today's summary: 'date', 'high', 'low', 'average', 'max_uv_index',
              'total_snow_cm', 'sun_hours', 'sunrise', 'sunset', 'moon_phase'.
            - 'hourly' (list[dict]): Today's forecast in 3-hour blocks, each with temperature,
              precipitation chances, wind, UV, and 'is_upcoming' (True if the block is at or
              after the current local hour, None if local time is unavailable).
            - 'upcoming_days' (list[dict]): Forecasts for the following days, each with
              'date', 'high', 'low', 'uv_index', 'total_snow_cm', and 'sun_hours'.

    Raises:
        Exception: If wttr.in returns a non-200 status or the location can't be found.
    '''
    url = f'https://wttr.in/{location}?format=j1'
    response = requests.get(url, timeout = 10)
    if response.status_code != 200:
        raise Exception(f'wttr.in error: status {response.status_code}')
    try:
        data = response.json()
    except ValueError:
        raise Exception(f'Could not find a location named {location}')

    current = data['current_condition'][0]
    area = data['nearest_area'][0]
    today = data['weather'][0]
    astronomy = today['astronomy'][0]

    local_time, timezone, local_hour = get_local_time(location)

    def format_hour(raw_time):
        hour = int(raw_time) // 100
        return f'{hour:02d}:00'

    hourly = []
    for hour in today['hourly']:
        hourly.append({
            'time': format_hour(hour['time']),
            'temperature': float(hour['tempF']),
            'feels_like': float(hour['FeelsLikeF']),
            'description': hour['weatherDesc'][0]['value'].strip(),
            'chance_of_rain': int(hour['chanceofrain']),
            'chance_of_snow': int(hour['chanceofsnow']),
            'chance_of_thunder': int(hour['chanceofthunder']),
            'chance_of_sunshine': int(hour['chanceofsunshine']),
            'cloud_cover': int(hour['cloudcover']),
            'humidity': int(hour['humidity']),
            'wind_speed': float(hour['windspeedMiles']),
            'wind_gust': float(hour['WindGustMiles']),
            'wind_direction': hour['winddir16Point'],
            'uv_index': int(hour['uvIndex']),
            'precipitation_inches': float(hour['precipInches']),
            'is_upcoming': None if local_hour is None else int(hour['time']) // 100 >= local_hour
        })

    upcoming = []
    for day in data['weather'][1:]:
        upcoming.append({
            'date': day['date'],
            'high': float(day['maxtempF']),
            'low': float(day['mintempF']),
            'uv_index': int(day['uvIndex']),
            'total_snow_cm': float(day['totalSnow_cm']),
            'sun_hours': float(day['sunHour'])
        })

    return {
        'location': location,
        'city': area['areaName'][0]['value'],
        'region': area['region'][0]['value'],
        'country': area['country'][0]['value'],


        'local_time': local_time,
        'timezone': timezone,
        'observation_time_utc': current['observation_time'],
        'temperature': float(current['temp_F']),
        'feels_like': float(current['FeelsLikeF']),
        'description': current['weatherDesc'][0]['value'].strip(),
        'humidity': int(current['humidity']),
        'cloud_cover': int(current['cloudcover']),
        'wind_speed': float(current['windspeedMiles']),
        'wind_direction': current['winddir16Point'],
        'uv_index': int(current['uvIndex']),
        'precipitation_inches': float(current['precipInches']),
        'visibility_miles': float(current['visibilityMiles']),
        'pressure': float(current['pressure']),

        'date': today['date'],
        'high': float(today['maxtempF']),
        'low': float(today['mintempF']),
        'average': float(today['avgtempF']),
        'max_uv_index': int(today['uvIndex']),
        'total_snow_cm': float(today['totalSnow_cm']),
        'sun_hours': float(today['sunHour']),

        'sunrise': astronomy['sunrise'],
        'sunset': astronomy['sunset'],
        'moon_phase': astronomy['moon_phase'],

        'hourly': hourly,
        'upcoming_days': upcoming
    }

SYSTEM_PROMPT = '''
You are a helpful assistant that recommends what to wear and which outdoor activities suit today's weather in a given city.

You have access to the get_current_weather tool. It takes a location, which can be a city on its own (e.g. Syracuse) or a city with a state or country (e.g. Syracuse, New York; Rome, Italy).

READING THE TOOL OUTPUT
- `local_time` and `timezone` are the real clock time in that city right now. This is "now" — anchor everything to it.
- `observation_time_utc` is UTC, not local time. Never show it to the user and never describe it as the local time.
- The top-level readings (temperature, feels_like, description, wind_speed, humidity, uv_index, precipitation_inches) are the most recent observation. Use these for current conditions rather than pulling an entry out of `hourly`.
- `hourly` covers the whole calendar day in local time, so most entries are already in the past. Each entry carries `is_upcoming`. Use only entries where `is_upcoming` is true when describing what is still to come, and ignore the rest.
- Compare `local_time` against `sunset`. If the day is nearly over, say so plainly and shift your suggestions to the evening and to `upcoming_days` instead of laying out a full day of activities.

Never present a time that has not arrived yet as if it were the current moment.

UNRECOGNIZABLE LOCATION
If you cannot recognize a city in what the user sent, do not call the tool and do not use the response format below. Reply with a single short paragraph and nothing else: no headings, no section titles, no bullet points. Say you could not identify a city in what they sent and ask for a recognizable city name, giving two or three examples of the form you accept. Stop there.

Never give clothing or activity advice without weather data. Generic layers, umbrellas and walking shoes for an unknown place are guesses dressed up as advice, and offering them is worse than saying nothing. The same applies if the tool fails or returns no usable data.

RESPONSE FORMAT
Use this format only when the tool returned weather data for a real location. In that case reply in exactly these four markdown sections, in this order, with these headings:

## Current Conditions
The city, the local time, and the latest readings: temperature and how it feels, sky conditions, wind, humidity, and the day's high and low. Two to four sentences.

## What to Wear
A markdown bullet list, three to five bullets, no paragraph text. Each bullet is one specific item or layer for walking out the door now, with the reason drawn from the numbers. Tie layering to how the temperature moves across the remaining hours. Give rain gear, sun protection, or wind protection their own bullet only when the numbers justify it.

## Outdoor Activities That Work Well Today
Two to four activities that fit the remaining daylight and conditions. Name the best window in local clock times and note when daylight runs out.

## What to Be Cautious About
Real concerns drawn from the data: high UV, chances of rain or thunder, strong gusts, heat or cold, poor visibility, or a narrow window before sunset. If nothing stands out, say conditions are mild and name the one thing still worth a glance.

Use the actual numbers and stay concrete. Never invent a reading the tool did not return.
'''

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

agent = create_agent(
    model = 'openai:gpt-5-mini',
    tools = [get_local_time, get_current_weather],
    system_prompt = SYSTEM_PROMPT
)

# Show title and description.
st.title("Lab 6")

st.caption('Enter a city and an LLM will give you a recommendation of what to wear based on the current weather!')

st.divider()

typed_input = st.text_input('Enter a city...', placeholder = 'Syracuse, NY')
location = typed_input.strip() or 'Syracuse, NY'

if st.button('Get recommendation'):

    with st.spinner(f'Checking the weather in {location}...'):
        result = agent.invoke(
            {'messages': [{'role': 'user', 'content': f'What should I wear in {location}?'}]}
        )

    with st.chat_message('assistant'):
        st.write(result['messages'][-1].content)

    with st.expander('Behind the scenes'):
        for message in result['messages']:
            st.write(f"**{message.type}**")
            st.write(message.content)
            if message.type == 'ai' and message.tool_calls:
                st.write(message.tool_calls)

# 12a. How many messages are there, and what is each one?
# There are 4 messages, the first is the prompt with the users city, the second is the model's tool call, the third is the tool calls result, and the fourth is the output the model created with information from the tool call.

#12b. Which messages is the model asking for the weather tool? (Its content is empty, Where is the request?)
# The model is asking for the weather in the second message. 

#In Lab 5, which lines of your code created each of those messages?
# In lab 5, the first message was created in this line: user_input = typed_input.strip() or DEFAULT_LOCATION which got the user input from a text field, or used the defualt location. The second message was created in this line:    response = client.chat.completions.create(model='gpt-5-mini', messages=messages, tools = tools,tool_choice = 'auto') which was an API call to an LLM with the tools included. The third message was created in this line: results = get_current_weather(tool_query_string) which called the get_current_weather tool with the query string as the input. The final message was created by this line: stream = client.chat.completions.create(model = 'gpt-5-mini', messages = messages, stream = True) which was the final API call which used the output of the tool call.