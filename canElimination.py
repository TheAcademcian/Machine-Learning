""" Concept Learning: Candidate elimination """
import numpy as np

x = np.array([['Sunny','Warm','Normal','Strong','Warm','Same','Yes'], 
              ['Sunny','Warm','High','Strong','Warm','Same','Yes'], 
              ['Rainy','Cold','High','Strong','Warm','Change','No'], 
              ['Sunny','Warm','High','Strong','Cool','Change','Yes']])
r,c = x.shape
print("The number of rows and columns are: ", r,"and", c-1)
print("The daatset is: \n",x)


s = np.empty(c-1, dtype=object)
for i in range(c-1): s[i] = "\u03A6"
print("\n Most specific hypotheis: \n",s)


g = np.empty(c-1, dtype=object)
for i in range(c-1): g[i] = "?"
print("\n Most general hypotheis: \n",g)

for i in range(c-1): s[i]  = x[0,i] # Load content from the first row

Flag = 0
for i in range(1,r):
    #print("\nSpecific hypotheis: \n",s) 
    if x[i,c-1] == 'Yes':
        for j in range(c-1):
            if x[i,j] != s[j]:
                s[j] = '?'
        print("\nIntermediate specific hypothesis: \n",s)
        if Flag == 0:
            Flag =1
            continue
        else:
            r1 = np.shape(g)[0]
            for i1 in range(r1):
                for j1 in range(c-1):
                    if g[i1,j1] != x[i,j1] and g[i1,j1] != "?":
                        g = np.delete(g, i1, axis=0)
              
        
    else:
        for j in range(c-1):
            g1 = np.empty(c-1, dtype=object)
            if s[j] == '?':
                continue
            elif x[i,j] != s[j]:
                for l in range(c-1): g1[l] = "?" 
                g1[j] = s[j]
                g = np.vstack([g, g1])
        g = np.delete(g, (0), axis=0)
        print("\nIntermediate general hypothesis\n",g)
     
        

            
print("\nSpecific hypotheis: \n",s) 
print("\nGeneral hypotheis: \n",g)


    

        
    
















