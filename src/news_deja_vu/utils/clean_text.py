def clean_ocr_text(text, basic, remove_list):
    """
    Given 
    - string of text,
    - whether (True/False) to do only basic newline cleaning, and
    - the list of characters to remove (if basic=False),
    returns a tuple containing
    (1) the text after applying the desired cleaning operations, and
    (2) a list of integers indicating, for each character in original text,
        how many positions to the left that character is offset to arrive at cleaned text.
    When basic is False, also replaces 'é', 'ï', 'ﬁ', and 'ﬂ'.
    In all cases, hyphen-newline ("-\n") sequences are removed, lone newlines are
    converted to spaces, and sequences of consecutive newlines are kept unchanged
    in order to indicate paragraph boundaries.
    """
    # Code to deal with unwanted symbols
    cleaned_text = text.replace("-\n", "")
    if not basic:
      cleaned_text = cleaned_text.replace("é", "e").replace("ï", "i").replace("ﬁ", "fi").replace("ﬂ", "fl")
      cleaned_text = cleaned_text.translate({ord(x): '' for x in remove_list})
      
    # Code to deal with newline and double newline
    z = 0
    while z < (len(cleaned_text)-1):  # Check from the first to before last index
          if cleaned_text[z] == "\n" and cleaned_text[z+1] == "\n":
              z += 2
          elif cleaned_text[z] == "\n" and cleaned_text[z+1] != "\n":
              temp = list(cleaned_text)
              temp[z] = " "
              cleaned_text = "".join(temp)
              z += 1
          else:
              z += 1
    if cleaned_text[len(cleaned_text)-1] == "\n" and cleaned_text[len(cleaned_text)-2] != "\n":  # Check if the last index is a new line
      temp = list(cleaned_text)
      temp[len(cleaned_text)-1] = " "
      cleaned_text = "".join(temp)  

    # Code to adjust offsets  
    offsets = []
    cur_offset = 0
    i = 0
      
    while i < len(text):
      if i+1 < len(text) and text[i:i+2] == '-\n':  # Found removed hyphen-newline
        offsets.extend([cur_offset, cur_offset + 1])  # Make removed characters correspond with next character
        cur_offset += 2  # Update offset for subsequent characters
        i += 2  # Push forward beyond removed characters
      else:
        offsets.append(cur_offset)  # Record offset of non-removed character
        i += 1  # Process next character
    
    if not basic:            
      for j in range(len(text)):
        if text[j] == "ﬁ" or text[j] == "ﬂ": 
            for a in range(j+1,len(text)):
                offsets[a] = offsets[a] - 1  # Negative offsets for every char after a char replaced with two chars
        elif text[j] in remove_list:
            for a in range(j+1,len(text)):  # Positive offsets for every char after a removed char
                offsets[a] = offsets[a] + 1
        else:
            j += 1
          
    return cleaned_text, offsets