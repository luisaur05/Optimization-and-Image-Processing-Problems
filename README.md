# Robust Conjugate Gradient Method for Non-smooth Convex Optimization and Image Processing Problems

## Overview
This project implements the Robust Conjugate Gradient Method (RCGM) to solve non-smooth convex optimization problems, specifically focusing on image processing tasks such as image restoration. The method uses Moreau-Yosida transformations to handle non-smoothness in the objective functions.

The RCGM is tested on two main tasks:
1. **Optimization of functions**: Demonstrating the effectiveness of the method on various non-smooth convex functions.
2. **Image restoration**: A practical application where we blur an image and then reconstruct it using the RCGM method.

## Requirements
- Python 3.x
- NumPy
- SciPy
- Matplotlib
- Pillow (for image processing)

## Installation

To install the required dependencies, use the following command:

```bash
pip install numpy scipy matplotlib pillow
```

## Files
- **`Optimizacion.ipynb`**: The main Jupyter notebook where the RCGM implementation and examples are shown.
- **`Imagen.png`**: The original image used in the image processing tasks.
- **`Imagen_resized.png`**: The resized image used for processing.
- **`datos.txt`**: The file containing intermediate results during the optimization process.
- **`Gradientes.txt`**: Contains gradient data used for plotting the behavior of the gradient during optimization.

## Key Functions

### `valor_absoluto(x)`
This function computes the absolute value of `x`:

```python
def valor_absoluto(x):
    if x > 0:
        return x
    return -x
```

### `maximo(x)`
This function computes `max(0, x)` for a given `x`:

```python
def maximo(x):
    if x > 0:
        return x
    return np.float64(0)
```

### `moreau_yosida_transform(f, x, lambda_)`
This function implements the Moreau-Yosida transformation of a function `f`:

```python
def moreau_yosida_transform(f, x, lambda_):
    if type(x) == cupy.ndarray:
        X = x.get()
    else:
        X = x
    def objective(y):
        return f(y) + (1 / (2 * lambda_)) * np.linalg.norm(X - y)**2
    result = minimize(objective, X, method='BFGS')
    return cp.array(result.fun)
```

### `gradient_conjugate_robust(F, grad_F, f, x0, lambd, gamma, sigma, c, epsilon)`
The core function implementing the Robust Conjugate Gradient Method:

```python
def gradient_conjugate_robust(F, grad_F, f, x0, lambd, gamma, sigma, c, epsilon):
    x_k = x0
    g_k = grad_F(f, x_k, lambd)
    d_k = -g_k
    archivo = open("datos.txt", "w")
    for i in range(1000):
        g_knorm = cp.linalg.norm(g_k)
        if g_knorm <= epsilon:
            break
        # Armijo line search
        alpha_k = 1.0
        while F(f, x_k + alpha_k * d_k, lambd) > F(f, x_k, lambd) + sigma * alpha_k * cp.dot(g_k, d_k):
            alpha_k *= c
        x_k1 = x_k + alpha_k * d_k
        archivo.write(str(x_k1))
        archivo.write("\n")
        g_k1 = grad_F(f, x_k1, lambd)
        s_k = x_k1 - x_k
        y_k = g_k1 - g_k
        y_knorm = cp.linalg.norm(y_k)
        s_knorm = cp.linalg.norm(s_k)
        d_knorm = cp.linalg.norm(d_k)
        t = alpha_k
        lastnumerador = (1 + t**2) * y_knorm**2 * (cp.dot(d_k, g_k1))
        lastdenominador = 4 * c * ((1 + t**2) * y_k - t * s_k).T @ g_k1
        if lastdenominador == 0:
            last = 0
        else:
            last = lastnumerador / lastdenominador
        T_k = max(gamma * d_knorm * y_knorm, d_knorm * s_knorm, cp.abs(cp.dot(d_k, y_k)), last)
        beta_k = (y_k - ((y_knorm**2) / (4 * c)) * ((d_k) / (T_k)) - ((t) / (1 + t**2)) * s_k).T * g_k1 / T_k
        d_k = -g_k1 + beta_k * d_k
        x_k = x_k1
        g_k = g_k1
    return x_k
```

## Example Usage

Here’s an example of using the robust conjugate gradient method to solve an image processing task:

```python
x0 = cp.array([3.0])
lambd = 1.0
gamma = 0.9
sigma = 0.1
c = 0.5
epsilon = 1e-20
solution = gradient_conjugate_robust(moreau_yosida_transform, grad_moreau_yosida_transform, valor_absoluto, x0, lambd, gamma, sigma, c, epsilon)
print("Solution:", solution)
```

### Image Restoration Example

For image restoration, after applying a blurring function, the `gradient_conjugate_robust` method is used to reconstruct the original image from the blurred version.

```python
image = Image.open('Imagen.png')
image = image.resize((s, s))
image = np.array(image)
imagen = np.ndarray((s, s))

for i in range(len(image)):
    for j in range(len(image[0])):
        imagen[i][j] = image[i][j][1]
        
# Process the image and perform optimization...
```

## Results

The final output displays:
1. The original image.
2. The processed image after applying the RCGM.
3. The degraded (blurred) image.

These results are displayed side-by-side for comparison.

## Conclusion

This project demonstrates the application of the Robust Conjugate Gradient Method to both theoretical convex optimization functions and practical image processing problems. The method proves to be effective for both tasks, showing a solid approach to solving non-smooth optimization problems.

