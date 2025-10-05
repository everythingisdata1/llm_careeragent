import json
import os

import gradio as gr
import requests
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion

load_dotenv(override=True)
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN", "your_pushover_key_here")
PUSHOVER_USER = os.getenv("PUSHOVER_USER", "your_pushover_user_here")
PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def push(message):
    try:
        response = requests.post(
            url=PUSHOVER_URL,
            json={
                "token": PUSHOVER_TOKEN,
                "user": PUSHOVER_USER,
                "message": message})
        if response.status_code == 200:
            return "Message pushed successfully!"
        else:
            return f"Failed to push message. Status code: {response.status_code}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


def record_user_details(email, name="Name not provided", notes="Notes not provided"):
    """This function records user details such as email, name, and notes. and works as LLM tools function"""
    push(f"Recording {name}'s details. Email: {email}, Notes: {notes}")
    return {"status": "success", "message": f"Recorded details for {name} with email {email}."}


def record_unknown_question(question):
    """This function records unknown questions asked by users and works as LLM tools function"""
    push(f"Recording unknown question: {question}")
    return {"status": "success", "message": f"Recorded unknown question: {question}."}


"""Json schema for the LLMs tools functions"""
record_user_details_json = {
    "name": "record_user_details",
    "description": "Records user details such as email, name, and notes.",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The user's email address."},
            "name": {"type": "string", "description": "The user's name."},
            "notes": {"type": "string", "description": "Additional notes about the user."}
        }
    }
}
record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Records unknown questions asked by users.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The unknown question asked by the user."}}
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]


class CareerAgent:
    """ LLM Career Agent"""

    def __init__(self, name):

        print("Initialize career agent  ")
        self.openai_client = OpenAI()
        self.name = name
        self.role = "Career Agent"
        try:
            self.reader = PdfReader("./me/Linkdin_bharatsingh.pdf")
            self.linkedIn = ""
            for page in self.reader.pages:
                self.linkedIn += page.extract_text()
            with open("./me/summary.txt", "r", encoding="utf-8") as f:
                self.summery = f.read()
        except Exception as err:
            print(err.with_traceback())

    def handle_tool_call(self, tool_calls):
        """This function is for calling LLMs tool"""
        print(f"Call Handle_tool call ::{tool_calls}")
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments
            print(f"Tool called:: {tool_name}")
            tool = globals().get(tool_name)
            result = tool(**arguments)
            print(f"Result :: {result}")
            results.append({
                "role": tool,
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results

    def system_prompt(self):
        system_prompt = (f"You are acting as {self.name}. You are answering questions on {self.name}'s website, "
                         f"as particularly questions is to represent {self.name} for interactions on the  website as faithfully as possible. "
                         f"You are given a summery of ({self.name}) background and linkedIn profile  which you can use to answer the questions. "
                         f"Be professional and engaging, as if talking on a potential client or future employer who cam across the website. "
                         f"If you don't knwo the answer of the question use your record_unkown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career.  "
                         f"If the user is engaging in discussion, try to steer them towards getting in touch via email: ask for their email and record it using your  record_user_details tool")
        system_prompt += f"\n\n ## Summery:: {self.summery} \n\n ## LinkedIn Profile:: \n{self.linkedIn}\n\n"
        system_prompt += f" With this context, please chat with the user, always staying in character as {self.name}"

        return system_prompt

    def chat(self, message, history):
        print("Invoke:: Chat with")
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        response: ChatCompletion = None
        while not done:
            response = self.openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages,tools=tools)
            if response.choices[0].finish_reason == 'tool_calls':
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls=tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content


if __name__ == '__main__':
    """This is Career Agent for LLM user """
    career_agent = CareerAgent(name="Bharat Singh")
    print(career_agent.chat)
    print("Initialization completed ")
    gr.ChatInterface(career_agent.chat, type="messages", show_api=True ).launch(share=True)
