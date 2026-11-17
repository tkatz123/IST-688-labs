import asyncio, os, time
import requests
import streamlit as st
from agents import Agent, Runner, function_tool

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
MODEL = "gpt-5-mini"

# ---------------------------------------------------------------------------
# Part B: weather function copied unchanged from Lab 5
# ---------------------------------------------------------------------------

def get_local_time(location):
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


@function_tool
def get_weather(location: str = "Syracuse, NY") -> dict:
    """Get the current weather and today's forecast for a location from wttr.in.

    Accepts a city on its own (e.g. "Syracuse") or a city with a state or country
    (e.g. "Syracuse, NY" or "Lima, Peru").

    Returns the resolved city, region and country; the real local time and timezone;
    current readings (temperature and feels-like in °F, sky description, humidity,
    cloud cover, wind speed in mph and direction, UV index, precipitation, visibility);
    today's high, low, sunrise and sunset; an hourly forecast for today in 3-hour
    blocks, where `is_upcoming` marks blocks that have not happened yet; and the
    highs and lows for the next few days. Fails if the location cannot be found.
    """
    return get_current_weather(location)  # the Lab 5 function


# ---------------------------------------------------------------------------
# Part C: the team
# ---------------------------------------------------------------------------

weather_agent = Agent(
    name="Weather agent",
    instructions=(
        "You are a weather reporter. Always call get_weather for the location you are given, "
        "then summarize the facts it returns in a short plain-text report. Include: the resolved "
        "city, region and country; the local time (use `local_time` and `timezone`, never "
        "`observation_time_utc`); current temperature and feels-like; sky conditions; wind; "
        "humidity; UV index; today's high and low; sunset time; and how temperature, rain, snow "
        "and wind change over the rest of the day, using only hourly entries where `is_upcoming` "
        "is true. Use the actual numbers and never invent a reading. Give facts only: no "
        "clothing advice, no activity suggestions. If the tool fails, or the city it resolved "
        "to is clearly not the place you were asked about, reply with one sentence saying the "
        "location could not be found and nothing else."
    ),
    tools=[get_weather],
    model=MODEL,
)

stylist_agent = Agent(
    name="Stylist agent",
    instructions=(
        "You are a practical stylist. You receive a weather report for a city. Recommend what to "
        "wear right now as a markdown bullet list of three to five items. Each bullet names one "
        "specific item or layer and gives the reason, drawn from the numbers in the report. Tie "
        "layering to how the temperature changes over the rest of the day. Only add rain, sun or "
        "wind protection when the numbers justify it. Use only facts from the report and never "
        "invent a reading. Do not suggest activities."
    ),
    model=MODEL,
)

activity_agent = Agent(
    name="Activity agent",
    instructions=(
        "You are a local activity planner. You receive a weather report for a city. Suggest two "
        "to four activities that fit the conditions and the daylight that is left, naming the best "
        "window for each in local clock times and noting when daylight runs out. If the day is "
        "nearly over, say so and suggest evening options instead. Then list any real cautions "
        "from the data (high UV, rain or thunder chances, strong gusts, heat or cold, poor "
        "visibility). Use only facts from the report and never invent a reading. Do not give "
        "clothing advice."
    ),
    model=MODEL,
)

manager_agent = Agent(
    name="Manager",
    instructions=(
        "You coordinate a team that tells the user what to wear and do today in a city.\n"
        "1. If the user's message does not name a recognizable real city, do not call any "
        "tools. Reply with one short paragraph (no headings, no bullets) saying you could not "
        "identify a city and asking for one, with two examples such as Syracuse, NY or "
        "Rome, Italy. Stop there.\n"
        "2. Otherwise call weather_report with the city. If the report says the location could "
        "not be found, reply the same way as in step 1 and call no other tools.\n"
        "3. Otherwise call stylist_advice and activity_ideas, passing each the full weather "
        "report as input.\n"
        "4. Combine the results into exactly these markdown sections, in this order: "
        "## Current Conditions (two to four sentences from the weather report), "
        "## What to Wear (the stylist's bullets), "
        "## Outdoor Activities That Work Well Today (the activity planner's suggestions), "
        "## What to Be Cautious About (cautions from the activity planner and weather report). "
        "Keep the specialists' numbers and do not add advice of your own."
    ),
    tools=[
        weather_agent.as_tool(
            tool_name="weather_report",
            tool_description=(
                "Look up the current weather for a city and return a factual summary "
                "(no advice). Input: the city, e.g. 'Lima, Peru'."
            ),
        ),
        stylist_agent.as_tool(
            tool_name="stylist_advice",
            tool_description=(
                "Recommend what to wear. Input: the full weather report from weather_report."
            ),
        ),
        activity_agent.as_tool(
            tool_name="activity_ideas",
            tool_description=(
                "Suggest activities with time windows plus weather cautions. "
                "Input: the full weather report from weather_report."
            ),
        ),
    ],
    model=MODEL,
)


# ---------------------------------------------------------------------------
# Part E: one agent doing the whole job, using the Lab 6 system prompt
# ---------------------------------------------------------------------------

SINGLE_AGENT_PROMPT = '''
You are a helpful assistant that recommends what to wear and which outdoor activities suit today's weather in a given city.

You have access to the get_weather tool. It takes a location, which can be a city on its own (e.g. Syracuse) or a city with a state or country (e.g. Syracuse, New York; Rome, Italy).

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

single_agent = Agent(
    name="Single agent",
    instructions=SINGLE_AGENT_PROMPT,
    tools=[get_weather],
    model=MODEL,
)


@st.cache_resource
def get_event_loop():
    return asyncio.new_event_loop()


def run(agent, prompt):
    start = time.perf_counter()
    result = get_event_loop().run_until_complete(Runner.run(agent, prompt))
    return result, time.perf_counter() - start


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

DEFAULT_LOCATION = 'Syracuse, NY'

st.title("Lab 9 - The \"What to Wear\" Crew")
st.caption('A weather agent, a stylist and an activity planner, coordinated by a manager.')
st.divider()

mode = st.radio('Who should answer?', ['Team', 'Single agent'], horizontal = True)
chosen_agent = manager_agent if mode == 'Team' else single_agent

with st.form('city_form', border = False, enter_to_submit = True):
    typed_input = st.text_input('Enter a city...', placeholder = DEFAULT_LOCATION)
    submitted = st.form_submit_button('Get advice')

location = typed_input.strip() or DEFAULT_LOCATION

if submitted:
    if not typed_input.strip():
        st.caption(f'No city entered — showing {DEFAULT_LOCATION}.')

    working = 'The team is' if mode == 'Team' else 'The single agent is'
    with st.spinner(f'{working} working on {location}...'):
        result, seconds = run(chosen_agent, f"What should I wear and do today in {location}?")

    with st.chat_message("assistant"):
        st.write(result.final_output)

    # Part D: behind the scenes
    #
    # Observed on a run for "Syracuse, NY" (7 model requests, ~14.7k tokens):
    #
    # a. Which tools did the manager call, and in what order?
    #    weather_report first (input "Syracuse, NY"), then stylist_advice and activity_ideas.
    #
    # b. What did the manager send to the stylist: the full forecast or a summary? Who wrote it?
    #    A summary. The stylist never saw the raw JSON from get_weather. The weather agent's
    #    model read the JSON and wrote a plain-text report (location, local time, current
    #    readings, high/low, sunset, upcoming hours). The manager's model then copied that
    #    report into the `input` argument of stylist_advice (and activity_ideas), so the
    #    text was written by the weather agent and passed along by the manager.
    #
    # c. "Model requests" is higher than the number of tool calls. Where did the others happen?
    #    The expander only shows the manager's 3 tool calls. Each agent-as-tool is its own
    #    run with its own model calls, and their usage rolls up into this total:
    #      Manager:        3 (pick weather_report, pick stylist + activity, write final answer)
    #      Weather agent:  2 (call get_weather, then write the summary)
    #      Stylist agent:  1
    #      Activity agent: 1
    #    = 7. The weather agent's own get_weather call happens inside its sub-run, so it
    #    doesn't appear in result.new_items either.
    usage = result.context_wrapper.usage
    with st.expander("Behind the scenes"):
        st.write(f"Model requests: {usage.requests} · Tokens: {usage.total_tokens:,} · "
                 f"Time: {seconds:.1f}s · Answered by: {result.last_agent.name}")
        for item in result.new_items:
            if item.type == "tool_call_item":
                st.write(f"{item.agent.name} called {item.raw_item.name}: "
                         f"{item.raw_item.arguments}")
            elif item.type == "tool_call_output_item":
                st.write(f"result: {str(item.output)[:300]}")
            elif item.type == "handoff_output_item":
                st.write(f"handoff: {item.source_agent.name} -> {item.target_agent.name}")
