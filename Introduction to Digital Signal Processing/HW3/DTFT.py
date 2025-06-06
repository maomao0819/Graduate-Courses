import numpy as np
import matplotlib.pyplot as plt

def method_1():
    # Define the range of 𝜔 values
    𝜔 = np.linspace(-np.pi, np.pi, num=1000)

    # Define the magnitude of R(𝑒^(𝑗𝜔)) and W(𝑒^(𝑗𝜔))
    M = 4
    R = np.abs(np.exp(-1j * 𝜔 * M/2) * (np.sin(𝜔 * (M + 1)/2) / np.sin(𝜔/2)))
    W = 0.5 * R * (1 - np.exp(1j * 𝜔/M)) + 0.5 * R * (1 - np.exp(-1j * 𝜔/M))

    # Create a plot of the magnitude of R(𝑒^(𝑗𝜔)) and W(𝑒^(𝑗𝜔))
    plt.figure()
    plt.plot(𝜔, R, label='R(𝑒^(𝑗𝜔))')
    plt.plot(𝜔, W, label='W(𝑒^(𝑗𝜔))')
    plt.xlabel('𝜔')
    plt.ylabel('Magnitude')
    plt.title('Magnitude of R(𝑒^(𝑗𝜔)) and W(𝑒^(𝑗𝜔)) for M = 4')
    plt.legend()
    plt.grid()
    plt.show()


def method_2():
    # Define the value of M
    M = 4

    # Define the DTFT of r[n]
    def R(w):
        # Compute the DTFT of r[n] for given w
        # using the geometric series summation formula
        return np.sum(np.exp(-1j * w * np.arange(M+1)))

    # Define the DTFT of w[n]
    def W(w):
        # Compute the DTFT of w[n] for given w
        # using the expression derived earlier
        X = np.cos(2 * np.pi * np.arange(M+1) / M)
        return 0.5 * (R(w) - X * np.exp(-1j * w * (np.arange(M+1) - 2 * np.pi / M)))

    # Generate frequency values from 0 to 2*pi
    w = np.linspace(0, 2 * np.pi, num=1000)

    # Compute the magnitude of R(e^(jw)) and W(e^(jw))
    mag_R = np.abs(R(w))
    mag_W = np.abs(W(w))

    # Plot the magnitude of R(e^(jw)) and W(e^(jw))
    plt.figure(figsize=(10, 6))
    plt.plot(w, mag_R, label='|R(e^(jw))|')
    plt.plot(w, mag_W, label='|W(e^(jw))|')
    plt.xlabel('Frequency (radians)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title('Magnitude of R(e^(jw)) and W(e^(jw)) for M = 4')
    plt.grid(True)
    plt.show()



def method_3():
    def r(n, M):
        return np.where((0 <= n) & (n <= M), 1, 0)

    def w(n, M):
        return 0.5 * (1 - np.cos(2 * np.pi * n / M)) * np.where((0 <= n) & (n <= M), 1, 0)

    M = 4
    n = np.arange(0, M+1)
    R = np.fft.fftshift(np.fft.fft(r(n, M)))
    W = np.fft.fftshift(np.fft.fft(w(n, M)))

    # Plotting the magnitude of R(e^(jw)) and W(e^(jw))
    w = np.linspace(-np.pi, np.pi, num=len(n))
    plt.figure()
    plt.plot(w, np.abs(R), label='R(e^(jw))')
    plt.plot(w, np.abs(W), label='W(e^(jw))')
    plt.xlabel('w')
    plt.ylabel('Magnitude')
    plt.title('Magnitude of R(e^(jw)) and W(e^(jw)) for M = 4')
    plt.legend()
    plt.grid()
    plt.show()

def method_4():

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

def method_5():
    # Define the DTFT of r[n] (R(e^(jw))) for M = 4
    def R(w):
        return np.sinc(w/(2*np.pi)) * np.exp(-1j * 3 * w/2)

    # Define the DTFT of w[n] (W(e^(jw))) for M = 4
    def W(w):
        return (1/2) * R(w) - (1/4) * R(w + 2*np.pi/4) - (1/4) * R(w - 2*np.pi/4)

    # Define the range of frequency values (w)
    w = np.linspace(-np.pi, np.pi, num=1000)

    # Calculate the magnitude of R(w) and W(w)
    R_w = np.abs(R(w))
    W_w = np.abs(W(w))

    # Plot the magnitude of R(w)
    plt.figure()
    plt.plot(w, R_w, label='|R(e^(jw))|')
    plt.xlabel('Frequency (w)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude of R(e^(jw)) vs. Frequency (w) for M = 4')
    plt.legend()
    plt.grid(True)

    # Plot the magnitude of W(w)
    plt.figure()
    plt.plot(w, W_w, label='|W(e^(jw))|')
    plt.xlabel('Frequency (w)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude of W(e^(jw)) vs. Frequency (w) for M = 4')
    plt.legend()
    plt.grid(True)

    plt.show()
method_2()