"""
Ethan McIlveen, February 2023

Simple script to implement the faq bot on a discord server

"""

import discord
import asyncio
from ml_faq_bot import *

##Create the client object
class MyClient(discord.Client):
    """Imported from discord, used to create the client object"""
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

    async def on_ready(self):
        print('Logged on as', self.user)

    async def on_message(self, message):
        # don't respond to ourselves
        if message.author == self.user:
            return

        # get the utterance and generate the response, will only work in a channel named 'bot-testing'
        if message.channel.name == "bot-testing":
            utterance = message.content
            response = understand(utterance)

        # send the response
        await message.channel.send(response)

client = MyClient()
# read the bot's token, then run the bot
with open("bot_token.txt") as file:
    token = file.read()

async def main():
    await client.start(token)

# Check if there's already a running event loop
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if str(e) == "asyncio.run() cannot be called from a running event loop":
            loop = asyncio.get_event_loop()
            loop.create_task(main())
        else:
            raise