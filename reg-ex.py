# regular_expressions.py
"""Volume 3: Regular Expressions.
<Name> Sophia Rawlings
<Class> Math 323 Section 2
<Date> February 13th 2020
"""
import re

# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #return a python pattern thing
    return re.compile("python")
    

# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #returns the exact string pattern they want us to with all the symbols
    return re.compile(r"\^\{\@\}\(\?\)\[\%\]\{\.\}\(\*\)\[\_\]\{\&\}\$")

# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"^(Book|Mattress|Grocery) (store|supplier)$")

# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    #matches exact python identifiers 
    return re.compile(r"^(([A-z]|_)+([0-9])*)+ *(= *[0-9]+|= *(([A-z]|_)+([0-9])*)+|= *'[^']*'|$)")

# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """
    #finds the code and matches it and then adds a colon to expressions needing colons
    pattern = re.compile(r"((for|if|elif|else|while|try|except|finally|with|def|class).*)", re.MULTILINE)
    new_code = pattern.sub(r"\1:", code) #bro how do you add colons correctly
    return new_code

# Problem 6
def prob6(filename="fake_contacts.txt"):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """
    big_dict = dict() #the dictionary that will have name and then the info
    small_dict = {"birthday" : None, "email": None, "phone": None} #the dictionary with the info
    
    name = re.compile(r"^([A-Z][a-zA-Z]*( [A-Z]\.)? [A-Z][a-zA-Z]*\b)") #pattern to find name
    email = re.compile(r"\S*(\S\@\w*(\.\w*))")                          #pattern to find emails
    phone_number = re.compile(r"(\d{3})\D*(\d{3})\D*(\d{4})\b")         #pattern to find phone num
    birthday = re.compile(r"(\d)?(\d)\/(\d)?(\d)\/(\d{2})?(\d{2})\b")  #pattern to find birthday
    
    with open(filename, 'r') as fp:
        line = fp.readline()
        #read through each line and find the info we are looking for
        while line:
            this_name = name.search(line).group(0)
            the_birthday = birthday.search(line)
            numba = phone_number.search(line)
            the_email = email.search(line)
            
            #if/else statements check to see if pattern objects are none or not
            if the_birthday == None:
                the_birthday = None    
            else:
                #sorry this is a very messy if/else for the birthdays 
                early_month = None
                early_day = None
                year_num = None
                if the_birthday.group(1) == None:
                    early_month = "0"
                if the_birthday.group(3) == None:
                    early_day = "0"
                if the_birthday.group(5) == None:
                    year_num = "20"
                    
                the_birthday = the_birthday.group(0)
                if early_month == "0" and early_day == "0" and year_num == "20":
                    the_birthday = re.sub(birthday,r"0\2/0\4/20\6",the_birthday)
                elif early_month == "0" and year_num == "20":
                    the_birthday = re.sub(birthday,r"0\2/\3\4/20\6",the_birthday)
                elif early_day == "0" and year_num == "20":
                    the_birthday = re.sub(birthday,r"\1\2/0\4/20\6",the_birthday)
                elif early_month == "0" and early_day == "0":    
                    the_birthday = re.sub(birthday,r"0\2/0\4/\5\6",the_birthday)
                elif early_month == "0":
                    the_birthday = re.sub(birthday,r"0\2/\3\4/\5\6",the_birthday)
                elif early_day == "0":
                    the_birthday = re.sub(birthday,r"\1\2/0\4/\5\6",the_birthday)
                elif year_num == "20":
                    the_birthday = re.sub(birthday,r"\1\2/\3\4/20\6",the_birthday)
                
            if numba == None:
                numba = None
            else:
                numbar = numba.group(0)
                numba = re.sub(phone_number,r"(\1)\2-\3",numbar)
            if the_email == None:
                the_email = None
            else:
                the_email = the_email.group(0)
                
            #input the info where it needs to be
            small_dict["birthday"] = the_birthday
            small_dict["email"] = the_email
            small_dict["phone"] = numba
            big_dict[this_name] = small_dict
            small_dict = {"birthday" : None, "email": None, "phone": None}
            line = fp.readline()    #iterate to the next line
    return big_dict
