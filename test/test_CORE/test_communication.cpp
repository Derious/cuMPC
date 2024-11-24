#include <emp-tool/emp-tool.h>
#include "../../MPABY_GMW/GMW_protocol.h"
// #include "../../MPABY_GC/GC_mpc.h"
using namespace std;
using namespace emp;

#define EMPDM_ReLU_comm
#define BENCH 1
const static int nP = 2;
int party, port;
const static int length = 1;

int main(int argc, char** argv) {
 
	parse_party_and_port(argv, &party, &port);
	printf("party:%d	port:%d\n",party,port);
	if(party > nP)return 0;

	NetIOMP<nP> io(party, port);
	ThreadPool pool(4);	
    PRG prg;


    int64_t* r_value = new int64_t[length];
    memset(r_value, 0, length * sizeof(int64_t));
    int64_t mask_input = 0;
    prg.random_data(r_value,length*sizeof(int64_t));

	GMWprotocolA<nP>* gmw = new GMWprotocolA<nP>(&io,&pool,party);



	#ifdef ReLU_comm
    if (party != 1) {
			io.recv_data(1, &mask_input, 64*sizeof(bool));
			io.flush(1);
		}
		else {
			vector<future<void>> res;
			for(int i = 2; i <= nP; ++i) {
				int party2 = i;
				res.push_back(pool.enqueue([&io, mask_input, party2]() {
					io.send_data(party2, &mask_input, 64*sizeof(bool));
					io.flush(party2);
				}));
			}
			joinNclean(res);
		}

	mpc->GMW_A->open(r_value,r_value);


    if (party == 2)
    {
        uint64_t band2 = io.count();
	    cout <<"ReLU bandwidth\t"<<party<<"\t"<<band2<<endl;
    }
	#endif

	#ifdef EMPDM_ReLU_comm


	auto start2 = clock_start();
	gmw->open_vec(r_value,r_value,length);
	
    
	// gmw->open_vec(r_value+length/2,r_value+length/2,length/2);
	double timeused2 = time_from(start2);
	cout << "Parallel time used: " << timeused2 / 1000 << " ms" << endl;

	auto start = clock_start();
	gmw->open_vec_2PC(r_value,r_value,length);
	
    
	// gmw->open_vec(r_value+length/2,r_value+length/2,length/2);
	double timeused = time_from(start);
	cout << "Sequential time used: " << timeused / 1000 << " ms" << endl;

	

	// int another_party = party == 1 ? 2 : 1;
	// auto start2 = clock_start();
	// if(party == 1){
	// 	gmw->io->send_data(another_party,r_value,length*sizeof(int64_t));
	// 	gmw->io->flush(another_party);
	// 	gmw->io->recv_data(another_party,r_value,length*sizeof(int64_t));
	// 	gmw->io->flush(another_party);
	// }
	// else{
	// 	gmw->io->recv_data(another_party,r_value,length*sizeof(int64_t));
	// 	gmw->io->flush(another_party);
	// 	gmw->io->send_data(another_party,r_value,length*sizeof(int64_t));
	// 	gmw->io->flush(another_party);
	// }
	// double timeused2 = time_from(start2);
	// cout << "Sequential time used: " << timeused2 / 1000 << " ms" << endl;

	
    // gmw->open_vec(r_value,r_value,length);
    if (party == 1)
    {
        uint64_t band2 = io.count();
	    cout <<"EMPDM_ReLU bandwidth\t"<<party<<"\t"<<band2<<endl;
    }
	#endif

	#ifdef TrunMUL_comm

	mpc->GMW_A->open(r_value,r_value);
	mpc->GMW_A->open(r_value,r_value);
    if (party == 1)
    {
        uint64_t band2 = io.count();
	    cout <<"TrunMUL_comm bandwidth\t"<<party<<"\t"<<band2<<endl;
    }
	#endif
    


}
