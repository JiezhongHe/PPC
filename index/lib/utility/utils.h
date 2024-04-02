#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <unistd.h>


#define _GNU_SOURCE
 
#include <sys/types.h>
#include <fcntl.h>
#include <malloc.h>

#include <errno.h>
#include <string.h>

#include "../configuration/config.h"

double get_time(timeval& st, timeval& et);
vector<vector<Label>> load_label_path(string file_name);

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

	Direct_IO_reader(string filename_, int buffer_size_);

	void read_file(char* buf, size_t size);

	bool is_file_end();

	~Direct_IO_reader();
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