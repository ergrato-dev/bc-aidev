# 📖 Glosario - Semana 24: RNNs

## B

### BPTT (Backpropagation Through Time)
Algoritmo para calcular gradientes en RNNs desplegando la red en el tiempo y aplicando backpropagation estándar.

### Bidirectional RNN
RNN que procesa la secuencia en ambas direcciones (forward y backward), capturando contexto pasado y futuro.

## C

### Cell State ($C_t$)
En LSTM, memoria a largo plazo que fluye a través de la red con modificaciones mínimas. Actúa como "cinta transportadora" de información.

## E

### Exploding Gradient
Problema donde los gradientes crecen exponencialmente durante BPTT, causando inestabilidad. Se mitiga con gradient clipping.

## F

### Forget Gate ($f_t$)
Puerta LSTM que decide qué información eliminar del cell state. Fórmula: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$

## G

### Gradient Clipping
Técnica para limitar la magnitud de los gradientes, previniendo exploding gradients. Típicamente `clip_grad_norm_(params, max_norm)`.

### GRU (Gated Recurrent Unit)
Variante simplificada de LSTM con 2 puertas (reset y update) en lugar de 4. Más eficiente computacionalmente.

## H

### Hidden State ($h_t$)
Estado interno de la RNN que se pasa entre pasos temporales. Contiene información resumida de la secuencia hasta el momento.

## I

### Input Gate ($i_t$)
Puerta LSTM que controla qué nueva información añadir al cell state. Fórmula: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$

## L

### LSTM (Long Short-Term Memory)
Arquitectura RNN con mecanismo de puertas que resuelve el problema de vanishing gradient, permitiendo aprender dependencias de largo plazo.

## M

### Many-to-Many
Arquitectura RNN donde cada entrada produce una salida (ej: traducción, etiquetado de secuencias).

### Many-to-One
Arquitectura RNN donde toda la secuencia produce una única salida (ej: clasificación de sentimiento).

## O

### One-to-Many
Arquitectura RNN donde una entrada produce una secuencia (ej: generación de texto a partir de imagen).

### Output Gate ($o_t$)
Puerta LSTM que controla qué parte del cell state se expone como hidden state. Fórmula: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$

## R

### Reset Gate ($r_t$)
Puerta GRU que decide cuánto del estado anterior olvidar. Fórmula: $r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$

### RNN (Recurrent Neural Network)
Red neuronal con conexiones recurrentes que permiten procesar secuencias manteniendo un estado interno.

## S

### Sequence-to-Sequence (Seq2Seq)
Arquitectura encoder-decoder para transformar secuencias (ej: traducción automática).

### Stacked RNN
RNN con múltiples capas apiladas, donde la salida de una capa es la entrada de la siguiente.

## T

### Truncated BPTT
Variante de BPTT que limita la propagación de gradientes a un número fijo de pasos para eficiencia.

## U

### Update Gate ($z_t$)
Puerta GRU que controla el balance entre estado anterior y candidato. Fórmula: $z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$

### Unrolling
Proceso de expandir una RNN en el tiempo para visualizar o calcular gradientes.

## V

### Vanishing Gradient
Problema donde los gradientes se reducen exponencialmente durante BPTT, impidiendo aprender dependencias de largo plazo. LSTM y GRU lo resuelven.
