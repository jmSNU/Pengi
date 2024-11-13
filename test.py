from wrapper import MSSWrapper 

mss_wrapper = MSSWrapper(config="base") #base or base_no_text_enc
audio_file_paths = ["dataset/mix0.wav", "dataset/mix1.wav"]
prompts = """
    This track includes a podcast dialogue with ocassional sound effects
    """
text_prompts = [prompts]*len(audio_file_paths)
print(text_prompts)

audio_prefix, audio_embeddings = mss_wrapper.get_audio_embeddings(audio_paths=audio_file_paths)
print(f"Audio embedding shape : {audio_embeddings.shape}")

text_prefix, text_embeddings = mss_wrapper.get_prompt_embeddings(prompts=text_prompts)
print(f"Audio embedding shape : {text_embeddings.shape}")


output = mss_wrapper.predict(
    audio_paths=audio_file_paths,
    text_prompts=text_prompts,
    audio_resample=True
)

print(f"Output shape : {output[0].shape}")