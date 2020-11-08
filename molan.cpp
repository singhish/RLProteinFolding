/**
 * molan.cpp Copyright 2009
 * Dr. Homayoun Valafar, Mikhail Simin
 *
 * This file is part of Redcraft.
 *
 * Redcraft is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Redcraft is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Redcraft.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "redcat.h"
#include "polypeptide.h"
#include "tensor.h"
#include <cstdlib>
#include <iostream>
#include <fstream>

#include <getopt.h>



using namespace std;
string program_name;
vector<string> mResidueNames;
Redcat *mRedcat = NULL;

unsigned int PARAM = 0;
#define INTO_PAF 1

void fillResidueNames(string mDataPath, string mRDCFilePrefix) {
	stringstream inName;
	inName << mDataPath << "/" << mRDCFilePrefix << ".1";
	ifstream in(inName.str().c_str());
	if (!in.is_open()) {
		cerr << "could not read from the rdc file " << inName.str() << " in Redcraft::fillResidueNames(). defaulting to 'ALA'" << endl;
		exit(1);
	}

	mResidueNames.clear();

	string sTrash;
	double nTrash;
	while (!in.eof()) {
		in >> sTrash;
		if (in.eof()) break;
		mResidueNames.push_back(sTrash); 
		for (int i = 0; i < 7; i++) {
			in.ignore(256, '\n');
		}
	}
}

void usage(string e = "") {
	if (e.compare("")) cerr << "Error: " << e << endl << endl;
	cerr << "Usage: " << program_name << " [options] \"string of phi psi angles\"" << endl;
	cerr << "\t-e\t\t: evaluate the structure fitness to RDC data provided by" << endl;
	cerr << "\t-d<string>\t: RDC data file prefix." << endl;
	cerr << "\t-p<string>\t: path to the RDC data (i.e. data/1brf) no trailing / character" << endl;
	cerr << "\t-m#\t\t: media count. Will look for files rdcPrefix.1 rdcPrefix.2 etc.." << endl;
	cerr << "\t-f<string>\t: when expecting output files specify this as prefix for redcat or pdb files" << endl;
	cerr << "\t-g\t\t: generate .pdb structure (must be used with -f)" << endl;
	cerr << "\t-o<number>\t: RDC mapping offset. Default: 0. Used when evaluating structure, or determining an Order Tensor" << endl;
	cerr << "\t-O\t\t: Print out the back-calculated Order Tensor(s)" << endl;
	cerr << "\t-D<string>\t: Print M-Distance. Expecting Sxx Syy Sxy Sxz Syz string(s)" << endl;
	cerr << "\t-c\t\t: Print out the back-calculated Residual Chemical Shift" << endl;
	cerr << "\t-r\t\t: Create .redcat.m# files per alignment media. (must be used with -f)" << endl;
	cerr << "\t-l\t\t: Print out cumulative lennard-jones Ca energy" << endl;
	cerr << "\t-L\t\t: Print out max lennard-jones Ca energy" << endl;
	cerr << endl;
	cerr << "\t--PAF\t\t: Rotate molecule into it's Principle Alignment Frame of the first alignment medium, as soon as it is computed." << endl;
	exit(EXIT_FAILURE);
}

template <class T> inline std::string toString(const T& t) {
	std::stringstream ss;
	ss << t;
	return ss.str();
}

vector<string> &split(const string &s, char delim, vector<string> &elems) {
	stringstream ss(s);
	string item;
	while (getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

vector<string> split(const string &s, char delim = ' ') {
	vector<string> elems;
	return split(s, delim, elems);
}

/*
vector<Tensor*> canonicalize_tensors(vector<Tensor*> inp) {
	int M = inp.size();
	vector<Tensor*> incanon;

	Matrix * rot = NULL; // Rotation from first alignment + second canon

	for (int m = 0; m < M; m++) {
		Tensor * fixed;
		if (m == 0) {
			// first alignment medium, get R
			Matrix S, R;
			inp[m]->Decompose(S, R);
			rot = new Matrix(R);
			fixed = new Tensor(S);
		} else {

			if (m == 1) {
				// second alignment medium, get canon matrix
				Matrix * canon = (
					rot->Transpose() *
					inp[m]->getMatrix() *
					*rot).canon();
				rot->operator =(*rot * *canon);
			}

			fixed = new Tensor(
				rot->Transpose() *
				inp[m]->getMatrix() * 
				*rot);
		}


		//cout << "Pushing:" << *(fixed) << endl;
		incanon.push_back(fixed);
	}
	delete rot;
	return incanon;
}
*/


int main(int argc, char *argv[]) {
	program_name = argv[0];
	bool evaluate = false;
	bool generate_structure = false;
	bool printOrderTensor = false;
	bool printOrderTensorDistance = false;
	bool redcatFiles = false;
	bool printEnergy = false;
	bool printMaxEnergy = false;
	bool printCSA = false;
	string dataPath = "", rdcFilePrefix = "", outputprefix, OTReference;
	int mMediaCount = 0, mOffset = 0;
	int c;

	if (argc <= 1) usage();
	bool skipLoop = false;
	string ddparam;
	
	/*
	 *tangles will receive the torsion angles as a string. The angles are always the last argument passed from the command line
	 *so here tangles is set equal to the last argument and then the last argument is cut off so it never reaches getopt()
	*/
	//string tangles(argv[argc-1]);
	string fname(argv[argc-1]);
	argc--;

	while((c = getopt(argc, argv, "Llep:d:m:f:go:OD:rc-:")) != -1)
		switch (c)
		{
			case 'L':
				printMaxEnergy = true;
				break;
			case 'l':
				printEnergy = true;
				break;
			case 'e': // evaluate structure fitness
				evaluate = true;
				break;
			case 'p': // RDC data file path
				dataPath = optarg;
				break;
			case 'd': // RDC data file prefix
				rdcFilePrefix = optarg;
				break;
			case 'm': // media count
				mMediaCount = atoi(optarg);
				break;
			case 'f': // output file prefix
				outputprefix = optarg;
				break;
			case 'g': // generate structure
				generate_structure = true;
				break;

			case 'o': // set offset
				mOffset = atoi(optarg);
				break;

			case 'O': // print out OrderTensor
				printOrderTensor = true;
				break;

			case 'D': // Order Tensor Distance
				printOrderTensorDistance = true;
				OTReference = optarg;
				break;

			case 'r': // create redcat files
				redcatFiles = true;
				break;
			case 'c': // print chemical shift
				printCSA = true;
				break;

			case '-': // doubledash --
				ddparam = string(optarg);
				if (ddparam == "PAF") PARAM |= INTO_PAF;
				break;
			

			default:
				usage();
}
        mRedcat = new Redcat(dataPath, rdcFilePrefix, mMediaCount);

	if (argv[1] == NULL) {
		cerr << "No values of torsion angles!" << endl;
		usage();
	}

	std::string line;
	std::ifstream infile(fname);
	while (std::getline(infile, line))
	{
		char tangles[line.length() + 1];
		strcpy(tangles, line.c_str());

		istringstream iss(tangles); // argv[1] is now the string of phi/psi
		double phi, psi;
		Polypeptide pp;
					pp.setVectors(mRedcat->getData(), mRedcat->gettotalDataCount(), mRedcat->getdataCount(), mRedcat->getshifts());

		bool mintor = false;
		
		if (dataPath.compare("") && rdcFilePrefix.compare("")) {
			fillResidueNames(dataPath, rdcFilePrefix);
		}

		vector<string> AATypes;
		string AAType = "";
		int aaid = mOffset;
		while (iss >> phi && iss >> psi && iss) {
			if (aaid >= mResidueNames.size()) AAType = "ALA";
			else AAType = mResidueNames[aaid];
			AATypes.push_back(AAType);
			pp.appendAminoAcid(phi, psi, AAType);
			aaid++;
			mintor = true;
		}
		if (!mintor) {
			usage("No values of torsion angles!");
		}
		pp.appendAminoAcid();
		int mResidueCount = pp.getResidueCount();

		/*

		 AFTER THIS LINE pp IS READY TO BE USED

		 */

		if (printEnergy) {
			cout << "Ca VDW: " << pp.vdwca() << endl;
		}
		if (printMaxEnergy) {
			cout << "Ca VDW: ";
			double vdw = -10e3;
			for (int i = 0; i < mResidueCount; i++) vdw = max(vdw, pp.LJDistance(i));
			cout << vdw << endl;
		}

		if ((evaluate || redcatFiles || printOrderTensor)
			&& !(dataPath.compare("") && rdcFilePrefix.compare("") && mMediaCount)) {
			usage("This operation requires data path, rdc file prefix and media count");
		}
		if (redcatFiles && !outputprefix.compare("")) usage("This operation requires output file prefix");

		if (dataPath.compare("") && rdcFilePrefix.compare("") && mMediaCount) {
			
			if (mOffset) mRedcat->offset(mOffset);

			if (PARAM & INTO_PAF) {
				cerr << "Rotating molecule into PAF" << endl;
				// rotate pp by S1's R to overlay MF on PAF
				Matrix * orderTensor = new Matrix(5, 1);
				mRedcat->calculateRMSD(pp, 0, orderTensor); // this will populate orderTensor with values

				Tensor T(*orderTensor);
				Matrix S(3, 3), R(3, 3);
				T.Decompose(S, R);
				Rotation PAF(R.Transpose());
				pp.rotate(PAF);
				pp.updateAtoms(0);
				delete orderTensor;
			}
		}


		if (generate_structure) {
			if (outputprefix.compare("")) {
				pp.writePDB(outputprefix + ".pdb", mOffset + 1);
			} else {
				usage("To generate a structure you must specify -f<prefix>");
			}
		}

		Matrix * ots[mMediaCount];

		double rdcfit;
		for (int m = 0; m < mMediaCount; m++) {
			if (redcatFiles) mRedcat->createRedcatFile(pp, (outputprefix + ".redcat.m" + toString(m + 1)).c_str(), m);

			ots[m] = new Matrix(5, 1);

			rdcfit = mRedcat->calculateRMSD(pp, m, ots[m]);

			if (evaluate) cout << rdcfit << " " << mRedcat->getReducedRDCCount(m, mResidueCount) << " ";
		}
		if (evaluate) cout << mRedcat->calculateRMSD(pp) << endl;

		cout << scientific;
		cout.precision(3);
		for (int m = 0; m < mMediaCount; m++) {
			if (printOrderTensor) {
				cout << "S" << (m + 1) << "  : \n" << Tensor(*(ots[m])); // no endl here printing matrix does it.
				//	Matrix S, R;
				//	Tensor(*ots[m]).Decompose(S, R);
				//	cout << "S" << (m+1) << "' : \n" << S; // no endl here printing matrix does it.
				//	cout << "R" << (m+1) << "\n" << R;
			}
		}

		if (printCSA) {
			double csa = 0;
			for (int m = 0; m < mMediaCount; m++) {
				Tensor * OT = new Tensor(*(ots[m]));
				for (int i = 0; i < pp.getResidueCount() - 1; i++) { // -1 for dummy at the end
					cout << "M" << m << " " << (i + mOffset + 1) << " N H C:";
					for (int a = 0; a < 3; a++) {
						csa = pp.getAminoAcid(i)->getCSAdd(a, OT);
						cout << "\t" << csa;
					}
					cout << endl;
				}
				delete OT;
			}
		}

		if (printOrderTensorDistance) {
			vector<string> RTP = split(toString(OTReference));
			vector<Tensor*> computed;
			vector<Tensor*> input;
			for (int m = 0; m < mMediaCount; m++) {
				Tensor * RT = new Tensor(
					atof(RTP[0+(m*5)].c_str()),
					atof(RTP[1+(m*5)].c_str()),
					atof(RTP[2+(m*5)].c_str()),
					atof(RTP[3+(m*5)].c_str()),
					atof(RTP[4+(m*5)].c_str())
					);
				computed.push_back(new Tensor(*ots[m]));
				input.push_back(RT);
			}
			double distance = multi_tensor_distance(computed, input);
			vector<Tensor*> cComputed = canonicalize_tensors(computed);
			vector<Tensor*> cInput = canonicalize_tensors(input);


			double otmDist = 0;
			for (int m = 0; m < mMediaCount; m++) {
				//cout << "Input " << m << ": " << endl << *(cInput[m]) << endl;
				//cout << "Compt " << m << ": " << endl << *(cComputed[m]) << endl;
				double dist = cInput[m]->otmDistance(*(cComputed[m]));
				otmDist += dist;
				cout << "d(" << m << ") = " << dist << endl;
				delete computed[m];
				delete input[m];
			}

			cout << "Total OT distance:" <<  otmDist << endl;
		}

		for (int m = 0; m < mMediaCount; m++) {
			if (ots[m]) delete ots[m];
		}
	}

	if (mRedcat) delete mRedcat;
	return 0;
}
