def asteriscos_al_mig(s):
    if len(s)%2==1:
        return s
    else:
        medio = int(len(s)/2)
        return s[:medio]+'**'+s[medio:]
