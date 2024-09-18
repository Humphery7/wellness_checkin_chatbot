import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import streamlit as st
import os


# print(f' here:{os.getcwd()}')
dir = os.path.join(os.getcwd(),'model_checkpoint')
output_dir = 'model_checkpoint'
max_len = 100



def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer



def generate_text(model_path, sequence, max_length):

    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')

    # Create an attention mask
    attention_mask = torch.ones(ids.shape, dtype=torch.long, device=ids.device)

    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))

def run_app():
    sequence2 = "[Q] I am feeling very sad, blaming myself for not being confident"
    max_len = 100
    generate_text(output_dir, sequence2, max_len)

# st.set_page_config(page_title="My App", page_icon=":guardsman:", layout="wide")
# st.markdown("""
#     <meta name="bypass-tunnel-reminder" content="true">
# """, unsafe_allow_html=True)
#
#
#
#
# st.title("💬 Wellness Checkin Chatbot")
#
# # if "messages" not in st.session_state:
# #     st.session_state["messages"] = ["How can I help you?"]
#
#
# st.chat_message("assitant").write("Hello, How can I be of help to you today")
#
# if prompt := st.chat_input(placeholder='please enter your message'):
#     # if not openai_api_key:
#     #     st.info("Please add your OpenAI API key to continue.")
#     #     st.stop()
#
#
#     # st.session_state.messages.append({"role": "user", "content": prompt})
#     st.session_state['messages'] = ['[Q]' +' '+ prompt]
#     st.chat_message("user").write(prompt)
#     response = generate_text(output_dir, prompt, max_len)
#     # msg = response.choices[0].message.content
#     # st.session_state.messages.append({"role": "assistant", "content": msg})
#     st.chat_message("assistant").write(response)
# #
#
if __name__ == "__main__":
    run_app()