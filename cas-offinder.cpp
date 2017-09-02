#include "config.h"

#include "cas-offinder.h"
#include "oclfunctions.h"

#include <sstream>
#include <iterator>
#include <cmath>

using namespace std;

vector<string> split(string const &input) {
    istringstream sbuffer(input);
    vector<string> ret((istream_iterator<string>(sbuffer)), istream_iterator<string>());
    return ret;
}

vector<string> split(string const &input, char delim) {
    istringstream sbuffer(input);
    vector<string> ret;
    string item;
    while (getline(sbuffer, item, delim)) ret.push_back(item);
	return ret;
}

inline unsigned char Cas_OFFinder::compbase(unsigned char base){
	if (base == 'A') return 'T';
	if (base == 'T') return 'A';
	if (base == 'G') return 'C';
	if (base == 'C') return 'G';
	if (base == 'R') return 'Y';
	if (base == 'Y') return 'R';
	if (base == 'M') return 'K';
	if (base == 'K') return 'M';
	if (base == 'H') return 'D';
	if (base == 'D') return 'H';
	if (base == 'B') return 'V';
	if (base == 'V') return 'B';
	return base;
}

void Cas_OFFinder::set_complementary_sequence(cl_char* seq, size_t seqlen) {
	size_t i, l = 0;
	cl_char tmp;

	for (i = 0; i < seqlen; i++) {
		seq[i] = compbase(seq[i]);
	}
	for (i = 0; i < seqlen / 2; i++) {
		tmp = seq[i];
		seq[i] = seq[seqlen - i - 1];
		seq[seqlen - i - 1] = tmp;
	}
}

void Cas_OFFinder::set_seq_flags(int* seq_flags, const cl_char* seq, size_t seqlen) {
	int i, n = 0;
	for (i = 0; i < seqlen; i++) {
		if (seq[i] != 'N') {
			seq_flags[n] = i;
			n++;
		}
	}
	if (i != n)
		seq_flags[n] = -1;
}

void Cas_OFFinder::initOpenCL(vector<int> dev_ids) {
	unsigned int i, j;

	cl_device_id* found_devices = new cl_device_id[MAX_DEVICE_NUM];
	cl_uint device_cnt;
	unsigned int platform_maxdevnum;

    unsigned int dev_id = 0;
    vector<cl_device_id> devices;

	for (i = 0; i < platform_cnt; i++) {
			cerr << "platform i" << i << std::endl;
	    oclGetDeviceIDs(platforms[i], m_devtype, MAX_DEVICE_NUM, found_devices, &device_cnt);
        for (j = 0; j < device_cnt; j++) {
					cerr << "device j" << j << std::endl;
            if (dev_ids.size() == 0 || (dev_ids.size() > 0 && find(dev_ids.begin(), dev_ids.end(), dev_id) != dev_ids.end()))
                devices.push_back(found_devices[j]);
            dev_id += 1;
        }
	}
    m_devnum = devices.size();

	if (m_devnum == 0) {
		cerr << "No OpenCL devices found." << endl;
		exit(1);
	}

	cl_context context;
	cl_program program;

	const size_t src_len = strlen(program_src);
	for (i = 0; i < m_devnum; i++) {
		// Create completely separate contexts per device to avoid unknown errors
		context = oclCreateContext(0, 1, &devices[i], 0, 0);
		m_contexts.push_back(context);
		program = oclCreateProgramWithSource(context, 1, &program_src, &src_len);
		oclBuildProgram(program, 1, &devices[i], "", 0, 0);
        if (m_devtype == CL_DEVICE_TYPE_CPU) {
		    m_finderkernels.push_back(oclCreateKernel(program, "finder_cpu"));
		    m_comparerkernels.push_back(oclCreateKernel(program, "comparer_cpu"));
        } else {
						cerr << "WARNING -OOPS- CFD scoring version of CasOFFinder only implemented for CPUs so far!!" << endl;
						exit(1);
            m_finderkernels.push_back(oclCreateKernel(program, "finder"));
            m_comparerkernels.push_back(oclCreateKernel(program, "comparer"));
        }
		m_queues.push_back(oclCreateCommandQueue(m_contexts[i], devices[i], 0));
		MAX_ALLOC_MEMORY.push_back(0);
		oclGetDeviceInfo(devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &MAX_ALLOC_MEMORY[i], 0);
		cerr << "devnum " << i << " MAX_ALLOC_MEMORY " << MAX_ALLOC_MEMORY[i] << endl;
	}
	delete[] found_devices;
	cerr << "Total " << m_devnum << " device(s) found." << endl;
}

Cas_OFFinder::Cas_OFFinder(cl_device_type devtype, string devarg) {
    unsigned int i, j;
    int step;
    vector<int> dev_ids;

    m_devtype = devtype;
    vector<string> id_args = split(devarg, ',');
    vector<string> id_indices;
    for (i = 0; i < id_args.size(); i++) {
        id_indices = split(id_args[i], ':');
        if (id_indices.size() == 1) {
            dev_ids.push_back(atoi(id_indices[0].c_str()));
        }
        else if (id_indices.size() == 2 || id_indices.size() == 3) {
            step = 1;
            if (id_indices.size() == 3) step = atoi(id_indices[2].c_str());
            for (j = atoi(id_indices[0].c_str()); j < atoi(id_indices[1].c_str()); j += step) {
                dev_ids.push_back(j);
            }
        }
        else {
            cerr << "Something wrong with the device ID argument. Use all available devices instead..." << endl;
        }
    }
    initOpenCL(dev_ids);
}

Cas_OFFinder::~Cas_OFFinder() {
	unsigned int i;
	for (i = 0; i < m_finderkernels.size(); i++)
		oclReleaseKernel(m_finderkernels[i]);
	for (i = 0; i < m_comparerkernels.size(); i++)
		oclReleaseKernel(m_comparerkernels[i]);
	for (i = 0; i < m_devnum; i++) {
		oclReleaseCommandQueue(m_queues[i]);
		oclReleaseContext(m_contexts[i]);
	}
	clearbufvec(&m_patternbufs);
	clearbufvec(&m_patternflagbufs);
	clearbufvec(&m_comparebufs);
	clearbufvec(&m_compareflagbufs);
	clearbufvec(&m_entrycountbufs);

	clearbufvec(&m_pamscoresbufs);
	clearbufvec(&m_pamscoresindbufs);
	clearbufvec(&m_cfdscoresbufs);
	clearbufvec(&m_cfdscoresindbufs);
}

void Cas_OFFinder::setChrData() {
	unsigned int dev_index;

	m_chrdatasize = chrdata.size();
	m_totalanalyzedsize = 0;
	m_lasttotalanalyzedsize = 0;
	m_lastloci = 0;

	m_dicesizes.clear();
	clearbufvec(&m_chrdatabufs);
	clearbufvec(&m_flagbufs);
	clearbufvec(&m_locibufs);

	for (dev_index = 0; dev_index < m_devnum; dev_index++) {
		// I'm still trying to trace exactly is going on here with dicesize determination/impacts/importance
		// subdevice will have access to several shared memory buffers (will they be copied over and require space?)
		// subdevice function calls may also allocate local memory for local variables
		// I guess this is about dividing things up into chunks the devices can handle (still not following it though)
		size_t dicesize1 =
			(MAX_ALLOC_MEMORY[dev_index] - sizeof(cl_char)* (3 * m_patternlen - 1)
				- sizeof(cl_uint)* (2 * m_patternlen + 3) - sizeof(cl_ushort))
			/
				(4 * sizeof(cl_char) + 3 * sizeof(cl_uint) + 2 * sizeof(cl_ushort));

		size_t dicesize2 = (m_chrdatasize / m_devnum) + ((m_chrdatasize%m_devnum == 0) ? 0 : 1);
		size_t dicesize = (size_t)MIN( dicesize1, dicesize2 );
		cerr << "dicesize1 " << dicesize1 << " dicesize2 " << dicesize2 << endl;
		m_dicesizes.push_back( dicesize ); // No more than maximum allocation per device
		cerr << "Dicesize: " << m_dicesizes[dev_index] << endl;
		m_chrdatabufs.push_back(oclCreateBuffer(m_contexts[dev_index], CL_MEM_READ_ONLY, sizeof(cl_char)* (m_dicesizes[dev_index] + m_patternlen - 1), 0));
		m_flagbufs.push_back(oclCreateBuffer(m_contexts[dev_index], CL_MEM_WRITE_ONLY, sizeof(cl_char)* m_dicesizes[dev_index], 0));
		m_locibufs.push_back(oclCreateBuffer(m_contexts[dev_index], CL_MEM_WRITE_ONLY, sizeof(cl_uint)* m_dicesizes[dev_index], 0));

		oclSetKernelArg(m_finderkernels[dev_index], 0, sizeof(cl_mem), &m_chrdatabufs[dev_index]);
		oclSetKernelArg(m_finderkernels[dev_index], 4, sizeof(cl_mem), &m_flagbufs[dev_index]);
		oclSetKernelArg(m_finderkernels[dev_index], 6, sizeof(cl_mem), &m_locibufs[dev_index]);

		oclSetKernelArg(m_comparerkernels[dev_index], 0, sizeof(cl_mem), &m_chrdatabufs[dev_index]);
		oclSetKernelArg(m_comparerkernels[dev_index], 1, sizeof(cl_mem), &m_locibufs[dev_index]);
		oclSetKernelArg(m_comparerkernels[dev_index], 7, sizeof(cl_mem), &m_flagbufs[dev_index]);
	}
}

bool Cas_OFFinder::loadNextChunk() {
	if (m_totalanalyzedsize == m_chrdatasize)
		return false;

	unsigned int dev_index;
	unsigned long long tailsize;

	m_activedevnum = 0;
	m_worksizes.clear();
	m_lasttotalanalyzedsize = m_totalanalyzedsize;

	for (dev_index = 0; dev_index < m_devnum; dev_index++) {
		tailsize = m_chrdatasize - m_totalanalyzedsize;
		m_activedevnum++;
		if (tailsize <= m_dicesizes[dev_index]) {
			oclEnqueueWriteBuffer(m_queues[dev_index], m_chrdatabufs[dev_index], CL_TRUE, 0, (size_t)(sizeof(cl_char)* (tailsize + m_patternlen - 1)), (cl_char *)chrdata.c_str() + m_totalanalyzedsize, 0, 0, 0);
			m_totalanalyzedsize += tailsize;
			m_worksizes.push_back(tailsize);
#ifdef DEBUG
			cerr << "Worksize: " << m_worksizes[dev_index] << ", Tailsize: " << tailsize << endl;
#endif
			break;
		}
		else {
			oclEnqueueWriteBuffer(m_queues[dev_index], m_chrdatabufs[dev_index], CL_TRUE, 0, sizeof(cl_char)* (m_dicesizes[dev_index] + m_patternlen - 1), (cl_char *)chrdata.c_str() + m_totalanalyzedsize, 0, 0, 0);
			m_totalanalyzedsize += m_dicesizes[dev_index];
			m_worksizes.push_back(m_dicesizes[dev_index]);
#ifdef DEBUG
			cerr << "Worksize: " << m_worksizes[dev_index] << ", Tailsize: " << tailsize << endl;
#endif
		}
	}
	cerr << m_activedevnum << " devices selected to analyze..." << endl;

	return true;
}

void Cas_OFFinder::findPattern() {
	unsigned int dev_index;
	cl_uint zero = 0;
	for (dev_index = 0; dev_index < m_activedevnum; dev_index++) {
		const size_t worksize = (size_t)m_worksizes[dev_index];
		oclEnqueueWriteBuffer(m_queues[dev_index], m_entrycountbufs[dev_index], CL_TRUE, 0, sizeof(cl_uint), &zero, 0, 0, 0);
		oclEnqueueNDRangeKernel(m_queues[dev_index], m_finderkernels[dev_index], 1, 0, &worksize, 0, 0, 0, 0);
	}

	for (dev_index = 0; dev_index < m_activedevnum; dev_index++) {
		oclFinish(m_queues[dev_index]);
		m_locicnts.push_back(0);
		oclEnqueueReadBuffer(m_queues[dev_index], m_entrycountbufs[dev_index], CL_TRUE, 0, sizeof(cl_uint), &m_locicnts[dev_index], 0, 0, 0);
		if (m_locicnts[dev_index] > 0) {
			m_flags.push_back((cl_char *)malloc(sizeof(cl_char)* m_locicnts[dev_index]));
			oclEnqueueReadBuffer(m_queues[dev_index], m_flagbufs[dev_index], CL_TRUE, 0, sizeof(cl_char)*m_locicnts[dev_index], m_flags[dev_index], 0, 0, 0);

			m_mmcounts.push_back((cl_ushort *)malloc(sizeof(cl_ushort)* m_locicnts[dev_index] * 2)); // Maximum numbers of mismatch counts
			m_directions.push_back((cl_char *)malloc(sizeof(cl_char)* m_locicnts[dev_index] * 2));
			m_mmlocis.push_back((cl_uint *)malloc(sizeof(cl_uint)* m_locicnts[dev_index] * 2));
			m_sitescores.push_back((cl_float *)malloc(sizeof(cl_float)* m_locicnts[dev_index] * 2));

			m_mmlocibufs.push_back(oclCreateBuffer(m_contexts[dev_index], CL_MEM_WRITE_ONLY, sizeof(cl_uint)* m_locicnts[dev_index] * 2, 0));
			m_mmcountbufs.push_back(oclCreateBuffer(m_contexts[dev_index], CL_MEM_WRITE_ONLY, sizeof(cl_ushort)* m_locicnts[dev_index] * 2, 0));
			m_directionbufs.push_back(oclCreateBuffer(m_contexts[dev_index], CL_MEM_WRITE_ONLY, sizeof(cl_char)* m_locicnts[dev_index] * 2, 0));
			m_sitescorebufs.push_back(oclCreateBuffer(m_contexts[dev_index], CL_MEM_READ_WRITE, sizeof(cl_float)* m_locicnts[dev_index] * 2, 0));

			oclSetKernelArg(m_comparerkernels[dev_index], 2, sizeof(cl_mem), &m_mmlocibufs[dev_index]);
			oclSetKernelArg(m_comparerkernels[dev_index], 8, sizeof(cl_mem), &m_mmcountbufs[dev_index]);
			oclSetKernelArg(m_comparerkernels[dev_index], 9, sizeof(cl_mem), &m_directionbufs[dev_index]);
			oclSetKernelArg(m_comparerkernels[dev_index], 16, sizeof(cl_mem), &m_sitescorebufs[dev_index]);
		}
		else {
			m_flags.push_back(0);
			m_mmcounts.push_back(0);
			m_sitescores.push_back(0);
			m_directions.push_back(0);
			m_mmlocis.push_back(0);
			m_mmlocibufs.push_back(0);
			m_mmcountbufs.push_back(0);
			m_sitescorebufs.push_back(0);
			m_directionbufs.push_back(0);
		}
	}
}

void Cas_OFFinder::releaseLociinfo() {
	unsigned int dev_index;

	for (dev_index = 0; dev_index < m_activedevnum; dev_index++) {
		free((void *)m_mmcounts[dev_index]);
		free((void *)m_flags[dev_index]);
		free((void *)m_directions[dev_index]);
		free((void *)m_mmlocis[dev_index]);
		free((void *)m_sitescores[dev_index]);
	}
	m_directions.clear();
	m_mmlocis.clear();
	m_mmcounts.clear();
	m_locicnts.clear();
	m_sitescores.clear();
	clearbufvec(&m_mmlocibufs);
	m_flags.clear();
	clearbufvec(&m_mmcountbufs);
	clearbufvec(&m_directionbufs);
	clearbufvec(&m_sitescorebufs);
}

void Cas_OFFinder::indicate_mismatches(cl_char* seq, cl_char* comp) {
	unsigned int k;
	for (k = 0; k < m_patternlen; k++)
		if ((comp[k] == 'R' && (seq[k] == 'C' || seq[k] == 'T')) ||
			(comp[k] == 'Y' && (seq[k] == 'A' || seq[k] == 'G')) ||
			(comp[k] == 'K' && (seq[k] == 'A' || seq[k] == 'C')) ||
			(comp[k] == 'M' && (seq[k] == 'G' || seq[k] == 'T')) ||
			(comp[k] == 'W' && (seq[k] == 'C' || seq[k] == 'G')) ||
			(comp[k] == 'S' && (seq[k] == 'A' || seq[k] == 'T')) ||
			(comp[k] == 'H' && (seq[k] == 'G')) ||
			(comp[k] == 'B' && (seq[k] == 'A')) ||
			(comp[k] == 'V' && (seq[k] == 'T')) ||
			(comp[k] == 'D' && (seq[k] == 'C')) ||
			(comp[k] == 'A' && (seq[k] != 'A')) ||
			(comp[k] == 'G' && (seq[k] != 'G')) ||
			(comp[k] == 'C' && (seq[k] != 'C')) ||
			(comp[k] == 'T' && (seq[k] != 'T')))
			seq[k] += 32;
}

void
Cas_OFFinder::load_pam_scores(string const & infile){

	// setting some class variables now for later character-based indexing (fast lookups for pam and CFD scoring)
	cbeg_ = 'A';
	cend_ = 'Z';
	char_range_ = cend_ - cbeg_;
	nbases_ = 4;
	pamlen_ = 2;

	string line;
	vector<string> sline;
	ifstream fi(infile.c_str(), ios::in);
	if (!fi.good()){
		cerr << "file " << infile << " is not good!" << endl;
		exit(1);
	}

	size_t const arraylen(pow(nbases_,pamlen_));
	pamscores.resize(arraylen,0);
	cerr << "PAM scores (cl_float) arraylen " << arraylen << " arraysize " << sizeof(cl_float)*arraylen << endl;

	size_t const arrayleninds(pow(char_range_+1,pamlen_));
	pamscoresind.resize(arrayleninds,0);
	cerr << "PAM scores indexing (cl_short) arraylen " << arrayleninds << " arraysize " << sizeof(cl_short)*arrayleninds << endl;

	unsigned short scoreindex(0);
	while (getline(fi, line)) {
		if (line.empty()) continue;
		sline = split(line);
		unsigned char p2 = sline[0][0];
		unsigned char p3 = sline[1][0];
		if(
			(unsigned short)p2<cbeg_ ||
			(unsigned short)p2>cend_ ||
			(unsigned short)p3<cbeg_ ||
			(unsigned short)p3>cend_){
			cerr << "Bad PAM score file input: " << line << endl; exit(1);
		}
		float score = atof(sline[2].c_str());
		// this should be simpler/faster than using implicit/explicit hash functions or nested std::maps (?)
		unsigned short charind(
			((unsigned short)p2 - cbeg_) * char_range_ // positon 2 dimension
			+ (unsigned short)p3 - cbeg_ // position 3 dimension
		);

#ifdef DEBUG
		cerr
			<< "p2 " << p2 << " short is " << short(p2) <<  " - cbeg_ is " << short(p2) - cbeg_
			<< " p3 " << p3 << " short is " << short(p3) <<  " - cbeg_ is " << short(p3) - cbeg_
			<< " char_range_ is " << char_range_
			<< " adj array index " << charind
			<< " scoreindex " << scoreindex
			<< endl;
#endif

		pamscores[scoreindex] = score;
		pamscoresind[charind] = scoreindex++;

	}
	fi.close();
	cerr << "PAM scores loaded from " << infile << endl;
}

void
Cas_OFFinder::load_CFD_scores(string const & infile){

	// setting some class variables now for later character-based indexing (fast lookups for pam and CFD scoring)
	cbeg_ = 'A';
	cend_ = 'Z';
	char_range_ = cend_ - cbeg_;
	nbases_ = 4;
	cfdlen_ = 20;

	string line;
	vector<string> sline;
	ifstream fi(infile.c_str(), ios::in);
	if (!fi.good()){
		cerr << "file " << infile << " is not good!" << endl; exit(1);
	}

	// 1.0 default/initialized value corresponds to perfect matches
	// initialize unspecified mappings: Doench et al. omit complements (score 1) in their data file
	// note that ALL OTHER LETTERS will get a score of 1.0, i.e. no penalty (e.g. N, Y, R, etc)
	size_t const arraylen(cfdlen_*pow(nbases_,2)); // size is cfdlen_ * all two-basepair combinations (16)
	cfdscores.resize(arraylen,1.0);
	cerr << "CFD scores (cl_float) arraylen " << arraylen << " arraysize " << sizeof(cl_float)*arraylen << endl;

	size_t const arrayleninds(cfdlen_ * pow(char_range_+1,2));
	cerr << "CFD indexing (cl_short) arraylen " << arrayleninds << " arraysize " << sizeof(cl_short)*arrayleninds << endl;
	cfdscoresind.resize(arrayleninds,-1);

	unsigned short scoreindex(0);
	while (getline(fi, line)) {
		if (line.empty()) break;
		sline = split(line);
		unsigned char crRNA = sline[0][0];
		unsigned char DNA = sline[1][0];
		unsigned short pos = atoi(sline[2].c_str());
		// CFD scores are 1-indexed by convention
		pos-=1;
		if(pos<0 || pos>=cfdlen_){
			cerr << "Bad CFD score file input: " << line << endl; exit(1);
		}
		float score = atof(sline[3].c_str());
		if(crRNA=='U') crRNA='T';

		// the Doench scorefiles are expressed as complementary bases,
		// which is a useful and ideal expression of reality.
		// however here we convert to identities for speed/efficiency
		DNA = compbase(DNA);

		if(
			(unsigned short)crRNA<cbeg_ ||
			(unsigned short)crRNA>cend_ ||
			(unsigned short)DNA<cbeg_ ||
			(unsigned short)DNA>cend_){
			cerr << "Bad CFD score file input: " << line << endl; exit(1);
		}

		// simpler/faster than using hash functions or nested std::maps?
		unsigned short charind = (
			pos*char_range_*char_range_ // position dimension
			+ ((unsigned short)crRNA - cbeg_) * char_range_ // crRNA dimension
			+  (unsigned short)DNA - cbeg_ // DNA dimension
		);

#ifdef DEBUG
		cerr
			<< "p2 " << p2 << " short is " << short(p2) <<  " - cbeg_ is " << short(p2) - cbeg_
			<< " p3 " << p3 << " short is " << short(p3) <<  " - cbeg_ is " << short(p3) - cbeg_
			<< " char_range_ is " << char_range_
			<< " adj array index " << charind
			<< " scoreindex " << scoreindex
			<< endl;
#endif

		cfdscores[scoreindex]=score;
		cfdscoresind[charind] = scoreindex++;
	}
	fi.close();

	// initialize unspecified mappings: Doench et al. omit complements (score 1) in their data file
	// note that ALL OTHER LETTERS will get a score of 1.0, i.e. no penalty (e.g. N, Y, R, etc)
	unsigned char bases[] = {'A','C','G','T'};
	for(short pos(0); pos<cfdlen_; ++pos){
		for(short j(0); j<4; ++j){
			for(short k(0); k<4; ++k){
				unsigned short charind(
					pos*char_range_*char_range_ // position dimension
					+ ((unsigned short)(bases[j]) - cbeg_) * char_range_ // crRNA dimension
					+  (unsigned short)(bases[k]) - cbeg_ // DNA dimension
				);
				if(cfdscoresind[charind]!=-1) continue;
				cfdscores[scoreindex] = 1.0;
				cfdscoresind[charind] = scoreindex++;
			}
		}
	}

#ifdef DEBUG
	cerr << "CFD matrix score 1 T T " << cfdscores[ cfdscoresind[
		0*char_range_*char_range_ // position dimension
		+ ((unsigned short)'T' - cbeg_) * char_range_ // crRNA dimension
		+ (unsigned short)'T' - cbeg_ // DNA dimension
	]] << endl;

	cerr << "CFD matrix score 15 A C " << cfdscores[ cfdscoresind[
		14*char_range_*char_range_ // position dimension
		+ ((unsigned short)'A' - cbeg_) * char_range_ // crRNA dimension
		+ (unsigned short)'C' - cbeg_ // DNA dimension
	]] << endl;
#endif

}

// the goal is to get this CFD scoring happening in the multithreads,
// ideally integrated with the on-the-fly mismatch checking/filtering
// so that one can use CFD thresholds rather than mismatch thresholds
// as the primary run-time filter
// update: this function has subsequently also been implemented in the OpenCL comparer parallel code
float
Cas_OFFinder::calc_CFD_score(cl_char* seq, cl_char* comp){
	float score(0);

	score = pamscores[ pamscoresind[
		( (unsigned short)(seq[m_patternlen-2]) - cbeg_) * char_range_
		+ (unsigned short)(seq[m_patternlen-1]) - cbeg_
	] ];

	for (int k(0); k < m_patternlen-3; k++){
		score *= cfdscores[ cfdscoresind[
			k*char_range_*char_range_ // position dimension
			+ ((unsigned short)(comp[k]) - cbeg_) * char_range_ // crRNA dimension
			+  (unsigned short)(seq[k]) - cbeg_ // DNA dimension
		]];

#ifdef DEBUG
		cerr << k << " " << comp[k] << " " << seq[k] << " "
		<< cfdscores[k][comp[k]][seq[k]] << " "
		<< pscore << " "
		<< score << endl;
#endif
	}
	return score;
}

void Cas_OFFinder::compareAll(const char* outfilename) {
	unsigned int compcnt, i, j, dev_index;
	cl_uint zero = 0;

	cl_char *cl_compare = new cl_char[m_patternlen * 2];
	cl_int *cl_compare_flags = new cl_int[m_patternlen * 2];

	char *strbuf = new char[m_patternlen + 1]; strbuf[m_patternlen] = 0;

	for (compcnt = 0; compcnt < m_totalcompcount; compcnt++) {
		memcpy(cl_compare, m_compares[compcnt].c_str(), m_patternlen);
		memcpy(cl_compare + m_patternlen, m_compares[compcnt].c_str(), m_patternlen);
		set_complementary_sequence(cl_compare + m_patternlen, m_patternlen);
		set_seq_flags(cl_compare_flags, cl_compare, m_patternlen);
		set_seq_flags(cl_compare_flags + m_patternlen, cl_compare + m_patternlen, m_patternlen);

		for (dev_index = 0; dev_index < m_activedevnum; dev_index++) {
			if (m_locicnts[dev_index] <= 0) continue;
			oclEnqueueWriteBuffer(m_queues[dev_index], m_comparebufs[dev_index], CL_FALSE, 0, sizeof(cl_char) * m_patternlen * 2, cl_compare, 0, 0, 0);
			oclEnqueueWriteBuffer(m_queues[dev_index], m_compareflagbufs[dev_index], CL_FALSE, 0, sizeof(cl_int) * m_patternlen * 2, cl_compare_flags, 0, 0, 0);
			oclEnqueueWriteBuffer(m_queues[dev_index], m_entrycountbufs[dev_index], CL_FALSE, 0, sizeof(cl_uint), &zero, 0, 0, 0);

			oclFinish(m_queues[dev_index]);

			oclSetKernelArg(m_comparerkernels[dev_index], 6, sizeof(cl_ushort), &m_thresholds[compcnt]);
			oclSetKernelArg(m_comparerkernels[dev_index], 15, sizeof(cl_float), &m_scorethresholds[compcnt]);

			const size_t locicnts = m_locicnts[dev_index];
			oclEnqueueNDRangeKernel(m_queues[dev_index], m_comparerkernels[dev_index], 1, 0, &locicnts, 0, 0, 0, 0);
		}

		unsigned long long loci;

		char comp_symbol[2] = { '+', '-' };
		bool isfile = false;
		ostream *fo;
		if (strlen(outfilename) == 1 && outfilename[0] == '-') {
			fo = &cout;
		} else {
			fo = new ofstream(outfilename, ios::out | ios::app);
			isfile = true;
		}
		unsigned long long localanalyzedsize = 0;
		unsigned int cnt = 0;
		unsigned int idx;
//		float cfdscore(0);
		for (dev_index = 0; dev_index < m_activedevnum; dev_index++) {
			if (m_locicnts[dev_index] > 0) {
				oclFinish(m_queues[dev_index]);
				oclEnqueueReadBuffer(m_queues[dev_index], m_entrycountbufs[dev_index], CL_TRUE, 0, sizeof(cl_uint), &cnt, 0, 0, 0);
				if (cnt > 0) {
					oclEnqueueReadBuffer(m_queues[dev_index], m_mmcountbufs[dev_index], CL_FALSE, 0, sizeof(cl_ushort)* cnt, m_mmcounts[dev_index], 0, 0, 0);
					oclEnqueueReadBuffer(m_queues[dev_index], m_directionbufs[dev_index], CL_FALSE, 0, sizeof(cl_char)* cnt, m_directions[dev_index], 0, 0, 0);
					oclEnqueueReadBuffer(m_queues[dev_index], m_mmlocibufs[dev_index], CL_FALSE, 0, sizeof(cl_uint)* cnt, m_mmlocis[dev_index], 0, 0, 0);
					oclEnqueueReadBuffer(m_queues[dev_index], m_sitescorebufs[dev_index], CL_FALSE, 0, sizeof(cl_float)* cnt, m_sitescores[dev_index], 0, 0, 0);
					oclFinish(m_queues[dev_index]);
					for (i = 0; i < cnt; i++) {
						loci = m_mmlocis[dev_index][i] + m_lasttotalanalyzedsize + localanalyzedsize;

						strncpy(strbuf, (char *)(chrdata.c_str() + loci), m_patternlen);
						if (m_directions[dev_index][i] == '-') set_complementary_sequence((cl_char *)strbuf, m_patternlen);
// this is now done in the OpenCL comparer code
//						cfdscore = calc_CFD_score((cl_char*)strbuf, (cl_char*)m_compares[compcnt].c_str());
						indicate_mismatches((cl_char*)strbuf, (cl_char*)m_compares[compcnt].c_str());
						for (j = 0; ((j < chrpos.size()) && (loci >= chrpos[j])); j++) idx = j;

						if(m_sitescores[dev_index][i] < m_scorethresholds[compcnt] && m_mmcounts[dev_index][i] > 4) continue;
						//if (m_mmcounts[dev_index][i] > m_thresholds[compcnt]) continue;

						(*fo) << m_compares[compcnt]
							<< '\t' << chrnames[idx]
							<< '\t' << loci - chrpos[idx]
							<< '\t' << strbuf
							<< '\t' << m_directions[dev_index][i]
							<< '\t' << m_mmcounts[dev_index][i]
							<< '\t' << m_sitescores[dev_index][i]
							<< endl;
					}
				}
			}
			localanalyzedsize += m_worksizes[dev_index];
		}
		if (isfile) ((ofstream *)fo)->close();
	}
	delete [] strbuf;
	delete [] cl_compare;
	delete [] cl_compare_flags;
}

void Cas_OFFinder::init_platforms() {
	oclGetPlatformIDs(MAX_PLATFORM_NUM, platforms, &platform_cnt);
	if (platform_cnt == 0) {
		cerr << "No OpenCL platforms found. Check OpenCL installation!" << endl;
		exit(1);
	}
}
void Cas_OFFinder::print_usage() {
	unsigned int i, j;
		cout << endl << endl <<
		"*MOD* Cas-OFFinder-CFD (" << __DATE__ << ")" << endl <<
		" Experimental/Academic (J Ashworth)" << endl <<
		endl <<
	cout << "Forked from Cas-OFFinder v2.4" << endl <<
		"Copyright (c) 2013 Jeongbin Park and Sangsu Bae" << endl <<
		"Website: http://github.com/snugel/cas-offinder" << endl <<
		endl <<
		"Usage: cas-offinder {input_file} C[device_id(s)] {output_file} {pam_scores_file} {cfd_scores_file}" << endl <<
		"C: using CPUs (only option at the moment)" << endl <<
		endl <<
		"Example input file:" << endl <<
		"/var/chromosomes/human_hg19" << endl <<
		"NNNNNNNNNNNNNNNNNNNNNRG" << endl <<
		"GGCCGACCTGTCGCTGACGCNNN 5 0.2" << endl <<
		"CGCCAGCGTCAGCGACAGGTNNN 5 0.2" << endl <<
		"ACGGCGCCAGCGTCAGCGACNNN 5 0.2" << endl <<
		"GTCGCTGACGCTGGCGCCGTNNN 5 0.2" << endl <<
		endl <<
		"Available device list:" << endl;

	cl_device_id devices_per_platform[MAX_DEVICE_NUM];
	cl_uint device_cnt;
	cl_char devname[255] = { 0, };
    cl_char platformname[255] = { 0, };

    unsigned int cpu_id = 0, gpu_id = 0, acc_id = 0;
	for (i = 0; i < platform_cnt; i++) {
        oclGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 255, &platformname, 0);
		oclGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, MAX_DEVICE_NUM, devices_per_platform, &device_cnt);
		for (j = 0; j < device_cnt; j++) {
			oclGetDeviceInfo(devices_per_platform[j], CL_DEVICE_NAME, 255, &devname, 0);
			cout << "Type: CPU, ID: " << cpu_id++ << ", <" << devname << "> on <" << platformname << ">" << endl;
		}
		oclGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, MAX_DEVICE_NUM, devices_per_platform, &device_cnt);
		for (j = 0; j < device_cnt; j++) {
			oclGetDeviceInfo(devices_per_platform[j], CL_DEVICE_NAME, 255, &devname, 0);
			cout << "Type: GPU, ID: " << gpu_id++ << ", <" << devname << "> on <" << platformname << ">" << endl;
		}
		oclGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ACCELERATOR, MAX_DEVICE_NUM, devices_per_platform, &device_cnt);
		for (j = 0; j < device_cnt; j++) {
			oclGetDeviceInfo(devices_per_platform[j], CL_DEVICE_NAME, 255, &devname, 0);
			cout << "Type: ACCELERATOR, ID: " << acc_id++ << ", <" << devname << "> on <" << platformname << ">" << endl;
		}
	}
}

void Cas_OFFinder::readInputFile(const char* inputfile) {
	unsigned int dev_index;
	string pattern, line;
	vector<string> sline;
	cl_uint zero = 0;

	ifstream fi(inputfile, ios::in);
	if (!fi.good()) {
		exit(1);
	}

	if (!fi.eof())
		getline(fi, chrdir);
	if (chrdir[chrdir.length()-1] == '\r')
		chrdir = chrdir.substr(0, chrdir.length()-1);

	if (!fi.eof())
		getline(fi, pattern);
	if (pattern[pattern.length()-1] == '\r')
		pattern = pattern.substr(0, pattern.length()-1);

	transform(pattern.begin(), pattern.end(), pattern.begin(), ::toupper);
	while (getline(fi, line)) {
		if (line.empty()) break;
		if (line[line.length()-1] == '\r')
			line = line.substr(0, line.length()-1);
		sline = split(line);
		transform(sline[0].begin(), sline[0].end(), sline[0].begin(), ::toupper);
		m_compares.push_back(sline[0]);
		m_thresholds.push_back(atoi(sline[1].c_str()));
		m_scorethresholds.push_back(atof(sline[2].c_str()));
	}
	fi.close();

	m_totalcompcount = m_thresholds.size();
	m_patternlen = (cl_uint)(pattern.size());

	cl_char *cl_pattern = new cl_char[m_patternlen * 2];
	memcpy(cl_pattern, pattern.c_str(), m_patternlen);
	memcpy(cl_pattern + m_patternlen, pattern.c_str(), m_patternlen);
	set_complementary_sequence(cl_pattern+m_patternlen, m_patternlen);
	cl_int *cl_pattern_flags = new cl_int[m_patternlen * 2];
	set_seq_flags(cl_pattern_flags, cl_pattern, m_patternlen);
	set_seq_flags(cl_pattern_flags + m_patternlen, cl_pattern + m_patternlen, m_patternlen);

	for (dev_index = 0; dev_index < m_devnum; dev_index++) {
		m_patternbufs.push_back(oclCreateBuffer(m_contexts[dev_index], CL_MEM_READ_ONLY, sizeof(cl_char) * m_patternlen * 2, 0));
		m_patternflagbufs.push_back(oclCreateBuffer(m_contexts[dev_index], CL_MEM_READ_ONLY, sizeof(cl_int) * m_patternlen * 2, 0));
		oclEnqueueWriteBuffer(m_queues[dev_index], m_patternbufs[dev_index], CL_FALSE, 0, sizeof(cl_char) * m_patternlen * 2, cl_pattern, 0, 0, 0);
		oclEnqueueWriteBuffer(m_queues[dev_index], m_patternflagbufs[dev_index], CL_FALSE, 0, sizeof(cl_int) * m_patternlen * 2, cl_pattern_flags, 0, 0, 0);

		m_comparebufs.push_back(oclCreateBuffer(m_contexts[dev_index], CL_MEM_READ_ONLY, sizeof(cl_char) * m_patternlen * 2, 0));
		m_compareflagbufs.push_back(oclCreateBuffer(m_contexts[dev_index], CL_MEM_READ_ONLY, sizeof(cl_uint) * m_patternlen * 2, 0));

		m_entrycountbufs.push_back(oclCreateBuffer(m_contexts[dev_index], CL_MEM_READ_WRITE, sizeof(cl_uint), 0));
		oclEnqueueWriteBuffer(m_queues[dev_index], m_entrycountbufs[dev_index], CL_FALSE, 0, sizeof(cl_uint), &zero, 0, 0, 0);
		oclFinish(m_queues[dev_index]);

//		cerr << pamscores.size() << " " << cfdscores.size() << " " << pamscoresind.size() << " " << cfdscoresind.size() << endl;
		m_pamscoresbufs.push_back(   oclCreateBuffer(m_contexts[dev_index], CL_MEM_READ_WRITE, sizeof(cl_float) * pamscores.size(), 0));
		m_cfdscoresbufs.push_back(   oclCreateBuffer(m_contexts[dev_index], CL_MEM_READ_WRITE, sizeof(cl_float) * cfdscores.size(), 0));
		m_pamscoresindbufs.push_back(oclCreateBuffer(m_contexts[dev_index], CL_MEM_READ_WRITE, sizeof(cl_short) * pamscoresind.size(), 0));
		m_cfdscoresindbufs.push_back(oclCreateBuffer(m_contexts[dev_index], CL_MEM_READ_WRITE, sizeof(cl_short) * cfdscoresind.size(), 0));
		oclEnqueueWriteBuffer(m_queues[dev_index],   m_pamscoresbufs[dev_index],CL_FALSE,0,sizeof(cl_float)*pamscores.size(),&pamscores.at(0),0,0,0);
		oclEnqueueWriteBuffer(m_queues[dev_index],   m_cfdscoresbufs[dev_index],CL_FALSE,0,sizeof(cl_float)*cfdscores.size(),&cfdscores.at(0),0,0,0);
		oclEnqueueWriteBuffer(m_queues[dev_index],m_pamscoresindbufs[dev_index],CL_FALSE,0,sizeof(cl_short)*pamscoresind.size(),&pamscoresind.at(0),0,0,0);
		oclEnqueueWriteBuffer(m_queues[dev_index],m_cfdscoresindbufs[dev_index],CL_FALSE,0,sizeof(cl_short)*cfdscoresind.size(),&cfdscoresind.at(0),0,0,0);

		oclSetKernelArg(m_finderkernels[dev_index], 1, sizeof(cl_mem), &m_patternbufs[dev_index]);
		oclSetKernelArg(m_finderkernels[dev_index], 2, sizeof(cl_mem), &m_patternflagbufs[dev_index]);
		oclSetKernelArg(m_finderkernels[dev_index], 3, sizeof(cl_uint), &m_patternlen);
		oclSetKernelArg(m_finderkernels[dev_index], 5, sizeof(cl_mem), &m_entrycountbufs[dev_index]);

		oclSetKernelArg(m_comparerkernels[dev_index], 3, sizeof(cl_mem), &m_comparebufs[dev_index]);
		oclSetKernelArg(m_comparerkernels[dev_index], 4, sizeof(cl_mem), &m_compareflagbufs[dev_index]);
		oclSetKernelArg(m_comparerkernels[dev_index], 5, sizeof(cl_uint), &m_patternlen);
		oclSetKernelArg(m_comparerkernels[dev_index], 10, sizeof(cl_mem), &m_entrycountbufs[dev_index]);

		oclSetKernelArg(m_comparerkernels[dev_index], 11, sizeof(cl_mem), &m_pamscoresbufs[dev_index]);
		oclSetKernelArg(m_comparerkernels[dev_index], 12, sizeof(cl_mem), &m_pamscoresindbufs[dev_index]);
		oclSetKernelArg(m_comparerkernels[dev_index], 13, sizeof(cl_mem), &m_cfdscoresbufs[dev_index]);
		oclSetKernelArg(m_comparerkernels[dev_index], 14, sizeof(cl_mem), &m_cfdscoresindbufs[dev_index]);

    if (m_devtype != CL_DEVICE_TYPE_CPU) {
        oclSetKernelArg(m_finderkernels[dev_index], 7, sizeof(cl_char) * m_patternlen * 2, 0);
        oclSetKernelArg(m_finderkernels[dev_index], 8, sizeof(cl_int) * m_patternlen * 2, 0);
        oclSetKernelArg(m_comparerkernels[dev_index], 17, sizeof(cl_char) * m_patternlen * 2, 0);
        oclSetKernelArg(m_comparerkernels[dev_index], 18, sizeof(cl_int) * m_patternlen * 2, 0);
    }
  }

	delete[] cl_pattern;
	delete[] cl_pattern_flags;
}
