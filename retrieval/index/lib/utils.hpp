#pragma once
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

#define _GNU_SOURCE
 
#include <sys/types.h>
#include <fcntl.h>
#include <malloc.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#include "config.hpp"

using namespace std;


pthread_barrier_t barrier;

string mask_file_name = "";

vector<vector<Label>> load_label_path(string file_name){
	vector<vector<Label>> result;
	ifstream fin(file_name);
	char c;
	int label;
	string tmp;
	vector<Label> vec;
	while(fin>>c){
		if(c=='['){
			continue;
		}else if(c==',' || c==' '){
			if(tmp.size() > 0){
				vec.push_back(atoi(tmp.c_str()));
			}
			tmp = "";
		}else if(c==']'){
			if(tmp.size() > 0){
				vec.push_back(atoi(tmp.c_str()));
			}
			tmp = "";
			if(vec.size() > 0){
				result.push_back(vec);
			}
			vec.clear();
		}else{
			string s(1, c);
			tmp += s;
		}
	}
	// print
	// for(auto v : result){
	// 	cout<<"{";
	// 	for(auto c : v){
	// 		cout<<c<<", ";
	// 	}
	// 	cout<<"}"<<endl;
	// }
	return result;
}

// only support read files in sequence
class Direct_IO_reader{
public:
	char* read_buffer;
	int buffer_size;
	int fd;
	string filename;
	int ret;
	int page_size;
	int ptr;
	int global_read_offset;
	unsigned long long file_ptr;
	unsigned long long file_size;

	Direct_IO_reader(string filename_, int buffer_size_){
		filename = filename_;
		page_size = getpagesize();

		buffer_size = (buffer_size_/page_size)*page_size;
		ret = 0;

		// get file size
		ifstream fin(filename);
		fin.seekg(0, ios::end);
		file_size = fin.tellg();
		fin.close();

		fd = open(filename.c_str(), O_RDWR | O_DIRECT, 0644);
		if(fd < 0){
			cout<<"Failed to open file:"<<filename<<endl;
		}

		// allocat the buffer
		read_buffer = (char*)memalign(page_size, buffer_size);
		if(read_buffer == NULL){
			cout<<"Failed to allocat buffers"<<endl;
		}
		
		ptr = 0;
		// prefetch the data
		read(fd, read_buffer, buffer_size);
		file_ptr += buffer_size;
	}

	void read_file(char* buf, size_t size){
		int remaining_size = size;
		int offset = 0;
		while(remaining_size > 0){
			int remaining_buf_size = buffer_size-ptr;
			if(remaining_size >= remaining_buf_size){
				memcpy(buf+offset, read_buffer+ptr, remaining_buf_size);
				ptr = 0;
				read(fd, read_buffer, buffer_size);
				file_ptr += buffer_size;
				remaining_size -= remaining_buf_size;
				offset += remaining_buf_size;
			}else{
				memcpy(buf+offset, read_buffer+ptr, remaining_size);
				ptr += remaining_size;
				remaining_size = 0;
			}
		}
	}

	bool is_file_end(){
		if(file_ptr >= file_size){
			return true;
		}
		return false;
	}

	~Direct_IO_reader(){
		close(fd);
		delete [] read_buffer;
	}
};

// int write_temp_file(char* buffer,size_t length) {
//     int len=length;
//     char filename_template[]="/tmp/temp_file.XXXXXX";
//     int fd=mkstemp(filename_template);
//     unlink(filename_template);//Unlink the file, so it'll be removed when close
//     printf("Template file name:%s\n",filename_template);
//     write(fd,&len,sizeof(len));
//     write(fd,buffer,len);
//     return fd;
// }

// char* read_temp_file(int fd, size_t* length) {
//     char* buffer;
//     lseek(fd,0,SEEK_SET);
//     read(fd,length,sizeof(size_t));
//     buffer=(char*)malloc(*length);
//     read(fd,buffer,*length);
//     close(fd); // Temp file will be deleted
//     return buffer;
// }