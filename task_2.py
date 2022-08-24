#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt


# In[3]:


image = cv2.imread(r"C:\Users\rkann\OneDrive\Desktop\dog.webp")
cv2.imshow("Dog", image)
cv2.waitKey(0)


# In[4]:


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("New Dog", gray_image)
cv2.waitKey(0)


# In[5]:


inverted_image = 255 - gray_image
cv2.imshow("Inverted", inverted_image)
cv2.waitKey()


# In[6]:


blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)


# In[7]:


inverted_blurred = 255 - blurred
pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
cv2.imshow("Sketch", pencil_sketch)
cv2.waitKey(0)


# In[8]:


cv2.imshow("original image", image)
cv2.imshow("pencil sketch", pencil_sketch)
cv2.waitKey(0)


# In[ ]:




