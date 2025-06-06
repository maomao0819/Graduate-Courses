import numpy as np
import matplotlib.pyplot as plt



# Define the DTFT of r[n] (R(e^(jw)))
def R(w, M):
    return np.sinc(w/(2*np.pi)) * np.exp(-1j * (M-1) * w/2)

# Define the DTFT of w[n] (W(e^(jw)))
def W(w, M):
    return (1/2) * R(w, M) - (1/4) * R(w + 2*np.pi/M, M) - (1/4) * R(w - 2*np.pi/M, M)

# Define the range of frequency values (w)
w = np.linspace(-np.pi, np.pi, num=1000)

# Set the value of M (maximum value of n for r[n])
M = 10

# Calculate W(w)
W_w = W(w, M)

# Plot the real part of W(w)
plt.figure()
plt.plot(w, np.real(W_w), label='Real{W(e^(jw))}')
plt.xlabel('Frequency (w)')
plt.ylabel('Amplitude')
plt.title('Real part of W(e^(jw)) vs. Frequency (w)')
plt.legend()
plt.grid(True)

# Plot the imaginary part of W(w)
plt.figure()
plt.plot(w, np.imag(W_w), label='Imag{W(e^(jw))}')
plt.xlabel('Frequency (w)')
plt.ylabel('Amplitude')
plt.title('Imaginary part of W(e^(jw)) vs. Frequency (w)')
plt.legend()
plt.grid(True)

plt.show()

