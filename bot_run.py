import streamlit as st
import streamlit.components.v1 as components

BOT_ID = "5abf019b-430c-4f2d-a799-35bfe125dca9"
CLIENT_ID = "36245163-ad56-4c1e-bc3b-9535cbd47ec4"

st.title("🤖 Brand Support Agent")

# Embed Botpress webchat
botpress_html = f"""
<script src="https://cdn.botpress.cloud/webchat/v3.5/inject.js"></script>
<style>
  #webchat .bpWebchat {{
    position: unset;
    width: 100%;
    height: 100%;
    max-height: 100%;
    max-width: 100%;
  }}
  #webchat .bpFab {{
    display: none;
  }}
</style>

<div id="webchat" style="width: 100%; height: 600px;"></div>

<script>
  window.botpress.on("webchat:ready", () => {{
    window.botpress.open();
  }});
  window.botpress.init({{
    "botId": "{BOT_ID}",
    "configuration": {{
      "version": "v2",
      "botName": "Brand Support Agent",
      "botDescription": "",
      "website": {{}},
      "email": {{}},
      "phone": {{}},
      "termsOfService": {{}},
      "privacyPolicy": {{}},
      "color": "#3276EA",
      "variant": "solid",
      "headerVariant": "glass",
      "themeMode": "light",
      "fontFamily": "inter",
      "radius": 4,
      "feedbackEnabled": false,
      "footer": "[⚡ by Botpress](https://botpress.com/?from=webchat)",
      "soundEnabled": false,
      "proactiveMessageEnabled": false,
      "proactiveBubbleMessage": "Hi! 👋 Need help?",
      "proactiveBubbleTriggerType": "afterDelay",
      "proactiveBubbleDelayTime": 10
    }},
    "clientId": "{CLIENT_ID}",
    "selector": "#webchat"
  }});
</script>
"""

components.html(botpress_html, height=650)

