# Tài liệu tham khảo

## Thuật toán
1. J. Canny, "A Computational Approach to Edge Detection," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. 8, no. 6, pp. 679–698, 1986.
2. D. Marr, E. Hildreth, "Theory of Edge Detection," *Proc. Royal Society of London B*, vol. 207, pp. 187–217, 1980.
3. S. Di Zenzo, "A Note on the Gradient of a Multi-Image," *Computer Vision, Graphics, and Image Processing*, vol. 33, no. 1, pp. 116–125, 1986.
4. I. Sobel, G. Feldman, "A 3×3 Isotropic Gradient Operator for Image Processing," trình bày tại Stanford AI Project, 1968.
5. D. H. Douglas, T. K. Peucker, "Algorithms for the Reduction of the Number of Points Required to Represent a Digitized Line or its Caricature," *Cartographica*, vol. 10, no. 2, pp. 112–122, 1973.
6. S. Suzuki, K. Abe, "Topological Structural Analysis of Digitized Binary Images by Border Following," *CVGIP*, vol. 30, no. 1, pp. 32–46, 1985 (cv2.findContours).
7. A. M. Andrew, "Another Efficient Algorithm for Convex Hulls in Two Dimensions," *Information Processing Letters*, vol. 9, no. 5, pp. 216–219, 1979.
8. C. Tomasi, R. Manduchi, "Bilateral Filtering for Gray and Color Images," *Proc. ICCV*, 1998.
9. N. Otsu, "A Threshold Selection Method from Gray-Level Histograms," *IEEE Trans. Systems, Man, and Cybernetics*, vol. 9, no. 1, pp. 62–66, 1979.
10. J. Immerkær, "Fast Noise Variance Estimation," *Computer Vision and Image Understanding*, vol. 64, no. 2, pp. 300–302, 1996.
11. D. L. Donoho, I. M. Johnstone, "Ideal Spatial Adaptation by Wavelet Shrinkage," *Biometrika*, vol. 81, no. 3, pp. 425–455, 1994 (ngưỡng phổ quát $\sigma\sqrt{2\ln N}$).
12. J. E. Hopcroft, R. M. Karp, "An $n^{5/2}$ Algorithm for Maximum Matchings in Bipartite Graphs," *SIAM J. Computing*, vol. 2, no. 4, pp. 225–231, 1973.
13. H. W. Kuhn, "The Hungarian Method for the Assignment Problem," *Naval Research Logistics Quarterly*, vol. 2, pp. 83–97, 1955.

## Đánh giá
14. W. K. Pratt, *Digital Image Processing*, Wiley, 1978 (Figure of Merit).
15. D. Martin, C. Fowlkes, J. Malik, "Learning to Detect Natural Image Boundaries Using Local Brightness, Color, and Texture Cues," *IEEE TPAMI*, vol. 26, no. 5, pp. 530–549, 2004 (benchmark BSDS, khớp một-một, ODS/OIS).

## Giáo trình và phần mềm
16. R. C. Gonzalez, R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018.
17. Slide bài giảng "Chapter 2. Image Processing Methods (part 2)", trang 26: đề bài (`tai_lieu/`).
18. OpenCV 5.0: `imgproc/src/canny.cpp`, tài liệu `cv2.Canny`, `cv2.approxPolyDP`, `cv2.getStructuringElement`.
19. scikit-image 0.26: `skimage/feature/_canny.py`, `_canny_cy.pyx`.
20. SciPy 1.18: `scipy.ndimage`, `scipy.sparse.csgraph.maximum_bipartite_matching`, `scipy.optimize.linear_sum_assignment`.
