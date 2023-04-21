from gramformer import Gramformer
import streamlit as st
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(1212)

class GecTorDemo:

    def __init__(self):
        st.set_page_config(
            page_title="GecTor Demo",
            layout="wide"
            )
        self.examples = [
            "what be the reason for everyone leave the comapny",
            "He are moving here.",
            "I am doing fine. How is you?",
            "How is they?",
            "Matt like fish",
            "the collection of letters was original used by the ancient Romans",
            "We enjoys horror movies",
            "Anna and Mike is going skiing",
            "I walk to the store and I bought milk",
            " We all eat the fish and then made dessert",
            "I will eat fish for dinner and drink milk",
            ]

    @st.cache_resource(show_spinner=False)
    def load_gf(_self, model: int):
        """
            Load Gramformer model
        """
        gf = Gramformer(models=model, use_gpu=False)
        return gf
    
    def main(self):
        st.markdown("# Replicating Grammarly <img src=\"https://imgs.search.brave.com/EYFiTdavQZV6Nf2kfIm7zvJvzKisli4smRZPbjYZ-P8/rs:fit:667:667:1/g:ce/aHR0cHM6Ly9jYXBp/Y2hlLmNvbS9yYWls/cy9hY3RpdmVfc3Rv/cmFnZS9ibG9icy9l/eUpmY21GcGJITWlP/bnNpYldWemMyRm5a/U0k2SWtKQmFIQkJj/MmRDSWl3aVpYaHdJ/anB1ZFd4c0xDSndk/WElpT2lKaWJHOWlY/MmxrSW4xOS0tMDA0/MzMzY2IzN2I4NjEy/MzgyNzNjNjcwNGY3/OTc4N2NkYTUyOTkx/YS9HcmFtbWFybHku/cG5n\" alt=\"logo\" height=\"50px\" width=\"50px\" />", unsafe_allow_html=True)
        st.markdown("## GECToR – Grammatical Error Correction: Tag, Not Rewrite \n This project is PyTorch implementation of the following paper: \n> [GECToR – Grammatical Error Correction: Tag, Not Rewrite](https://aclanthology.org/2020.bea-1.16/)")

        with st.spinner('Loading model..'):
            gf = self.load_gf(1)
    
        input_text = st.selectbox(
            label="Choose an example",
            options=self.examples
            )
        input_text = st.text_input(
            label="Input text",
            value=input_text
        )

        if input_text.strip():
            results = gf.correct(input_text)
            results = list(results)
            corrected_sentence = results[0]
            st.markdown(f'#### Output:')
            st.write('')
            st.success(corrected_sentence)

        else:
            st.warning("Please select/enter text to proceed")
        
if __name__ == "__main__":
    obj = GecTorDemo()
    obj.main()