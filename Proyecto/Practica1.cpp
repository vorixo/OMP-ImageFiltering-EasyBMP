// Alvaro Jover Alvarez
// Jordi Amoros Moreno

// FILTRO MEDIANO + GAUSSIANO

#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <sys/time.h>
#include <omp.h>
#include "EasyBMP.h"

using namespace std;

// Constantes para filtro gaussiano
const double PI = 3.141592654;

// Numero maximo de hilos para la cpu actual
const int MAX_NUM_THREADS = omp_get_max_threads();

// Tamanyo de imagen
int WIDTH;
int HEIGHT;

// Struct para comparar
struct Compare {
	int valor;
	int indice;
};

// String es numero
bool is_digits(const std::string &str)
{
	return str.find_first_not_of("0123456789") == string::npos;
}

// Aplicar Gaussian Filter unidireccional
float Gaussian1D(int x, int sigma) {
	return exp(-x * x / (2 * sigma * sigma)) / sigma / sqrt(2 * PI);
}

void test() {
	#pragma omp parallel
	{
			int ID = omp_get_thread_num();
			cout << "Thread " << ID << " preparado..." << endl;
	}
}

int main() {
	// Variable imagen
	BMP Image;
	BMP Mediana;
	BMP gausshori,gaussverti;
	BMP Media;

	// Input para imagen
	string str;
	cout << "Introduce el nombre de la imagen: ";
	getline(cin, str);
	const char *nombreimagen = str.c_str();

	// Leemos la imagen
	if (Image.ReadFromFile(nombreimagen)) {

		// Variables no relevantes
		string toInt = "";
		bool fallo = false;

		// Datos de la imagen
		WIDTH = Image.TellWidth();
		HEIGHT = Image.TellHeight();
		Mediana.SetSize(WIDTH, HEIGHT);
		Mediana.SetBitDepth(Image.TellBitDepth()); // cambiar bitdepth al de input
		gausshori.SetSize(WIDTH, HEIGHT);
		gausshori.SetBitDepth(Image.TellBitDepth());
		gaussverti.SetSize(WIDTH, HEIGHT);
		gaussverti.SetBitDepth(Image.TellBitDepth());
		Media.SetSize(WIDTH, HEIGHT);
		Media.SetBitDepth(Image.TellBitDepth());

		cout << "Dimensiones imagen: " << WIDTH << " x " << HEIGHT << endl;

		// Variables de dependencia del algoritmo
		int dim = 0;

		// -----------------------------------------------------
		cout << "Introduce dimension de la malla (impar): ";

		//Pasar la entrada a int
		getline(cin, toInt);

		if (is_digits(toInt)) {
			istringstream ss(toInt);
			ss >> dim;

			if (dim % 2 == 0) {
				fallo = true;
			}
		}

		// Manejo de errores (cortamos la ejecucion)
		if (fallo) {
			cout << "Error: Introduce un numero impar." << endl;
			getchar();
			return 0;
		}

		// Numero de valores que contendra la malla
		int dimord = dim*dim;
		volatile int halfdimord = dimord / 2;
		int half = dim/2;

		// -----------------------------------------------------
		cout << "Fuerza del filtro [1-100] (Median Filter): ";
		int fuerza = 0;

		// Pasar la entrada a int
		getline(cin, toInt);

		if (is_digits(toInt)) {
			istringstream ss(toInt);
			ss >> fuerza;
		}

		// Manejo de errores (cortamos la ejecucion)
		if (fuerza > 100 || fuerza < 1) {
			cout << "Error: Valor " << fuerza << " fuera de rango para fuerza." << endl;
			getchar();
			return 0;
		}

		// SIGMA PARA GAUSSIANO
		cout << "Valor de sigma [1-50] (Gaussian Filter): ";
		int sigma = 1; // CAMBIAR PARA DIFERENTES VALORES DE SIGMA

		// Pasar la entrada a int
		getline(cin, toInt);

		if (is_digits(toInt)) {
			istringstream ss(toInt);
			ss >> sigma;
		}

		// Manejo de errores (cortamos la ejecucion)
		if (sigma > 50 || sigma < 1) {
			cout << "Error: Valor " << sigma << " fuera de rango para sigma." << endl;
			getchar();
			return 0;
		}

		int hilos;

		cout << "Numero de threads [1 - " << MAX_NUM_THREADS << "] : ";

		getline(cin, toInt);

		if (is_digits(toInt)) {
			istringstream ss(toInt);
			ss >> hilos;
		}

		if (hilos > MAX_NUM_THREADS) {
			cout << "Error: Valor " << MAX_NUM_THREADS << " fuera de rango." << endl;
			getchar();
			return 0;
		}

		omp_set_num_threads(hilos);
		// test(); Comprobante numero de hebras activas

		// -----------------------------------------------------
		// --  EMPEZAMOS OPERACIONES DE CPU SIN PARALELIZAR   --
		// -----------------------------------------------------

		// Variables t (tiempo monotonico para evitar doblar los ticks del procesador)
		struct timespec start, finish;
		double elapsed;

		// Mediana
		unsigned char neighboursred[dimord];
		unsigned char neighboursblue[dimord];
		unsigned char neighboursgreen[dimord];
		unsigned char ordenacion;

		// Gaussiano
		float gaussian_filter[dimord];
		float acumuladorgauss = 0;
		unsigned count;
		struct Compare maximored, maximoblue, maximogreen;

		clock_gettime(CLOCK_MONOTONIC, &start);

		// OpenMP

		/////////////////////////////
		//    FILTRO MEDIANA        //
		//////////////////////////////
		int tmp;
		int w, s, y, x, k, l;

		// For indicativo de la fuerza del algoritmo
		for (int i = 0; i < fuerza; i++) {

			// Recorremos la imagen
			#pragma omp parallel for \
		 	shared(Image, WIDTH, HEIGHT, dim, half) \
		 	private(y,x,w,s,k,l,tmp,count,neighboursred, neighboursblue, \
			neighboursgreen, maximored, maximoblue,maximogreen) \
			schedule(static)
			for (y = 0; y < WIDTH; y++) {
				for (x = 0; x < HEIGHT; x++) {

					// Guardamos los valores de la imagen en una malla de tamanyo dim*dim
					count = 0;

					for (w = 0; w < dim && (y + w - half) < WIDTH; w++) {
						for (s = 0; s < dim && (x + s - half) < HEIGHT; s++) {

							if (y + w - half >= 0 && x + s - half >= 0) {
								neighboursred[count] = Image(y + w - half, x + s - half)->Red;
								neighboursblue[count] = Image(y + w - half, x + s - half)->Blue;
								neighboursgreen[count] = Image(y + w - half, x + s - half)->Green;
								count++;
							}
						}
					}

					// Ordenamos para extraer la mediana
					for (k = count-1; k >= 0; --k) {
						maximored.valor = neighboursred[k];
						maximored.indice = k;

						maximoblue.valor = neighboursblue[k];
						maximoblue.indice = k;

						maximogreen.valor = neighboursgreen[k];
						maximogreen.indice = k;

						for (l = k-1; l >= 0; --l) {
							if (neighboursred[l] > maximored.valor) {
								maximored.valor = neighboursred[l];
								maximored.indice = l;
							}
							if (neighboursblue[l] > maximoblue.valor) {
								maximoblue.valor = neighboursblue[l];
								maximoblue.indice = l;
							}
							if (neighboursgreen[l] > maximogreen.valor) {
								maximogreen.valor = neighboursgreen[l];
								maximogreen.indice = l;
							}
						}
						tmp = neighboursred[k];
						neighboursred[k] = maximored.valor;
						neighboursred[maximored.indice] = tmp;

						tmp = neighboursblue[k];
						neighboursblue[k] = maximoblue.valor;
						neighboursblue[maximoblue.indice] = tmp;

						tmp = neighboursgreen[k];
						neighboursgreen[k] = maximogreen.valor;
						neighboursgreen[maximogreen.indice] = tmp;
					}



					// Mediana
					Mediana(y,x)->Red = neighboursred[count / 2];
					Mediana(y,x)->Blue = neighboursblue[count / 2];
					Mediana(y,x)->Green = neighboursgreen[count / 2];
				}
			}

		}

		// Median filter aplicado
		// Descomentar para ver
		// Mediana.WriteToFile("Mediana.bmp");
		cout << "Mediana completa, iniciando Gaussiano..." << endl;

		//////////////////////////////
		//    FILTRO GAUSIANO       //
		//////////////////////////////
		for (int i = 0; i < dimord; i++) {
			gaussian_filter[i] = Gaussian1D(abs(i - halfdimord), sigma);
			acumuladorgauss += gaussian_filter[i];
		}

		for (int i = 0; i < dimord; i++) {
			gaussian_filter[i] /= acumuladorgauss;
		}

		// Utilizaremos valores flotantes para el gaussiano
		float rojo, azul, verde;
		// int k;

		#pragma omp parallel private(x, y, k, rojo, verde, azul) shared(Image, gaussian_filter, dimord, gausshori, gaussverti)
    	{
			#pragma omp for
			for (y = 0; y < HEIGHT; y++) {
				for (x = 0; x < WIDTH; x++) {
					rojo = 0;
					azul = 0;
					verde = 0;
					for (k = -halfdimord; k <= halfdimord; k++) {
						if ((x + k >= 0) && (x + k < WIDTH)) {

							rojo += gaussian_filter[k + halfdimord] * Image(x + k, y)->Red;
							verde += gaussian_filter[k + halfdimord] * Image(x + k, y)->Green;
							azul += gaussian_filter[k + halfdimord] * Image(x + k, y)->Blue;
						}
					}
					gausshori(x, y)->Red = (ebmpBYTE)rojo;
					gausshori(x, y)->Green = (ebmpBYTE)verde;
					gausshori(x, y)->Blue = (ebmpBYTE)azul;
				}
			}

			#pragma omp for
			for (y = 0; y < WIDTH; y++) {
				for (x = 0; x < HEIGHT; x++) {
					rojo = 0;
					azul = 0;
					verde = 0;

					for (k = -halfdimord; k <= halfdimord; k++) {
						if ((x + k >= 0) && (x + k < HEIGHT)) {

							rojo += gaussian_filter[k + halfdimord] * gausshori(y,x+k )->Red;
							verde += gaussian_filter[k + halfdimord] * gausshori(y,x+k)->Green;
							azul += gaussian_filter[k + halfdimord] * gausshori(y,x+k)->Blue;
						}
					}
					gaussverti(y, x)->Red = (ebmpBYTE)rojo;
					gaussverti(y, x)->Green = (ebmpBYTE)verde;
					gaussverti(y, x)->Blue = (ebmpBYTE)azul;

				}
			}
		}

		// Descomentar para ver
		// gaussverti.WriteToFile("Gaussiano.bmp");
		cout << "Gaussiano completo, iniciando Media..." << endl;

		//////////////////////////////
		//      FILTRO MEDIA        //
		//////////////////////////////
		if(hilos > 2){
			#pragma omp parallel for private(y, x) \
			shared(WIDTH, HEIGHT, gaussverti, Mediana, Media) schedule(static)
			for (y = 0; y < WIDTH; y++) {
				for (x = 0; x < HEIGHT; x++) {
					Media(y, x)->Red = (gaussverti(y, x)->Red + Mediana(y, x)->Red) / 2;
					Media(y, x)->Blue = (gaussverti(y, x)->Blue + Mediana(y, x)->Blue) / 2;
					Media(y, x)->Green = (gaussverti(y, x)->Green + Mediana(y, x)->Green) / 2;
				}
			}
		} else {
			for (y = 0; y < WIDTH; y++) {
				for (x = 0; x < HEIGHT; x++) {
					Media(y, x)->Red = (gaussverti(y, x)->Red + Mediana(y, x)->Red) / 2;
					Media(y, x)->Blue = (gaussverti(y, x)->Blue + Mediana(y, x)->Blue) / 2;
					Media(y, x)->Green = (gaussverti(y, x)->Green + Mediana(y, x)->Green) / 2;
				}
			}
		}
		

		Media.WriteToFile("Media.bmp");

		clock_gettime(CLOCK_MONOTONIC, &finish);

		elapsed = (finish.tv_sec - start.tv_sec);
		elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;


		cout << "Tiempo de ejecucion para " << dim <<
			" de tamano de malla y " << fuerza << " de fuerza en la mediana a la resolucion dada: "
			<< elapsed << " segundos" << endl;

		cout << endl << "Pulsa intro para finalizar...";
	}

	else {
		cerr << endl << "CONSEJO: Asegurate que el archivo de entrada tiene el formato correcto (BMP),"
			<< " esta en el directorio correcto (../Sample) y esta escrito correctamente." << endl;
	}

	getchar();

	return 0;

}
