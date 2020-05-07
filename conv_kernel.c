/*
 * Copyright (C) 2015-2020 ETH Zurich and University of Bologna
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdint.h>
#include "pulp.h"

/*
  Inputs, weights and outputs are represented in fixed-point Q1.7 unsigned format:
  this means that each integer in [0-255] represents a real value in the range [0.0-1.0)
  The relationship between the integer I and real R representations is given by

     R = I * 2^-FRACTIONARY_BITS

 */
#define FRACTIONARY_BITS 7
#define ROUNDBIT   (1 << (FRACTIONARY_BITS -1))
#define SATURATION 255

// K = 3
void __attribute__ ((noinline)) ConvKxK_Naive  (uint8_t * In_Img, uint8_t * Out_Img, int R, int lb, int ub, int C, uint8_t  * Kernel, int K)
{
  int r, c, k, i, j, w, t;
  uint8_t coeff;
  uint8_t data;
  int S;

  //image board is black
  for (r=lb; r < ub; r++) {
    for (c=3/2; c < C-3/2; c++) {

        S = 0;
        t = r*R + c;
        //move in the window
        /* Coordinate window
            (-1;-1) (-1;0) (-1;+1)
            ( 0;-1) ( 0;0) ( 0;+1)
            (+1;-1) (+1;0) (+1;+1)
        */
        for (i = -K/2; i <= K/2; i++) {
            for (j = -K/2; j <= K/2; j++) {
                k = (r+i)*R + (c+j); //coeff for one dimension matrix
                data = In_Img[k];
                w = (i+1)*K + (j+1);
                coeff = Kernel[w];
                S = S + (int)(coeff*data);
            }
        }

        // Normalization: Data are Q2.2*(FRACTIONARY_BITS-1), now Q2.FRACTIONARY_BITS-1
        S = S >> FRACTIONARY_BITS;
        // Saturation
        S = S > SATURATION ? SATURATION : S;
        S = S <          0 ?          0 : S;

        Out_Img[t] = (uint8_t)(S);
    }
  }
}
