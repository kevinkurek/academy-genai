import streamlit as st
from openai import OpenAI
import weaviate
import os
import re

WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME")


def get_relevant_chunks(user_prompt, limit=5, max_distance=0.75):
    import weaviate.classes as wvc

    client = weaviate.connect_to_local(
        host="weaviate",
        port=8081,
        auth_credentials=weaviate.auth.AuthApiKey("adminkey"),
        headers={"X-OpenAI-Api-key": os.getenv("OPENAI_API_KEY")},
        skip_init_checks=True,
    )

    try:
        source = client.collections.get(WEAVIATE_COLLECTION_NAME)

        response = source.query.near_text(
            query=user_prompt,
            distance=max_distance,
            return_metadata=wvc.query.MetadataQuery(distance=True),
            limit=limit,
        )

    finally:
        client.close()

    print("Weaviate query response:", response)
    return response.objects


def get_response(chunks, user_prompt):
    inference_prompt = """You are the helpful social post generator Astra! You will create an interesting factoid post 
    about Apache Airflow and the topic requested by the user:"""

    for chunk in chunks:
        props = chunk.properties
        # print("CHUNK PROPS:", props)

        folder_path = props.get("folder_path") or "unknown"
        text = props.get("full_text") or props.get("text") or "no text"

        chunk_info = folder_path + " Full text: " + text
        inference_prompt += " " + chunk_info + " "

    inference_prompt += "Your user asks:"

    inference_prompt += " " + user_prompt

    inference_prompt += """ 
    Remember to keep the post short and sweet! At the end of the post add another sentence that is a space fact!"""

    client = OpenAI()

    champion_model_id = "gpt-4"

    chat_completion = client.chat.completions.create(
        model=champion_model_id,
        messages=[{"role": "user", "content": inference_prompt}],
    )

    return chat_completion


# ------------------ #
# STREAMLIT APP CODE #
# ------------------ #

st.title("My own use case!")

st.header("Search")

user_prompt = st.text_input(
    "Your post idea:",
    "Create a LinkedIn post for me about dynamic task mapping!",
)
limit = st.slider("Retrieve X most relevant chunks:", 1, 20, 5)
max_distance = st.slider("max_distance threshold for relevancy", 0.0, 1.0, 0.75)

if st.button("Generate post!"):
    st.header("Answer")
    with st.spinner(text="Thinking... :thinking_face:"):
        chunks = get_relevant_chunks(user_prompt, limit=limit, max_distance=max_distance)
        response = get_response(chunks=chunks, user_prompt=user_prompt)

        st.success("Done! :smile:")

        st.write(response.choices[0].message.content)

        st.header("Sources")

        for chunk in chunks:
            props = chunk.properties
            print("CHUNK PROPS:", props)  # to docker logs
            # st.write("Raw props:", props)

            title = props.get("title", "unknown title")
            folder_path = props.get("folder_path", "unknown folder")
            # raw_text = props.get("full_text") or props.get("text") or ""
            # clean_text = _clean_markdown_text(raw_text)
            # text_snippet = clean_text[:200]
            uuid = str(chunk.uuid)
            distance = getattr(chunk.metadata, "distance", None)

            st.write(f"Title: {title}")
            st.write(f"Folder: {folder_path}")
            st.write(f"UUID: {uuid}")
            st.write(f"Distance: {distance}")
            # st.write(f"Text snippet: {text_snippet}...")
            st.write("---")
