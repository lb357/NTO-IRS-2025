def sinput(text:str="", data:str="") -> str:
    input_data = input(text)
    if input_data == "":
        return data
    else:
        return input_data