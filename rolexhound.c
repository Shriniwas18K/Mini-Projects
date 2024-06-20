#include<stdio.h>
#include<stdint.h>
#include<stdbool.h>
#include<unistd.h>
#include<string.h>
#include<windows.h>
//contains apis for file monitoring,searching,metadata

#define EXT_SUCCESS           0
#define EXT_WITH_ERR          1
#define EXT_WITH_TOO_FEW_ARGS 1

/*
Purpose of program : This program is server that will continously monitor if any changes after 10s
occur in file given its path as shell arguement during initiation and , then send the 
notifications to client subscribed to that file
It is publish-subscribe event-driven micro-services architecture
*/

/*  Pre -requisites and Internal definations in libraries 

DWORD is datatype widely used in WINDOWS APIs , it is accessed using %lu  , i.e. unsigned long
WORD is also similarly integer kind of datatype like above and is accessed using %d
FILETIME can be converted to human-readable format using FileTimeToSystemTime(),but will add 
uneccessary complexity in codebase here so we will not do that , instead simly print it using %lu

typedef struct _WIN32_FILE_ATTRIBUTE_DATA {
  DWORD    dwFileAttributes;
  FILETIME ftCreationTime;
  FILETIME ftLastAccessTime;
  FILETIME ftLastWriteTime;
  DWORD    nFileSizeHigh;
  DWORD    nFileSizeLow;
} WIN32_FILE_ATTRIBUTE_DATA, *LPWIN32_FILE_ATTRIBUTE_DATA;
*/

/*  ACTUAL WORKFLOW OF THIS PROGRAM

first we will search for the file given as shell-arguement(cmd arguement to start execution)
that filePath is shell arguement , which gets extracted from the absolute path if given
and is stored into basePath

Now Server Starts

we will use functions FindFirstFile, GetFileAttributes and structs WIN32_FILE_DATA from windows.h

we will search wheter that file exists and store the results of the search in the hsearch variable
if it exists then we will get its attributes using GetFileAttributes(),which are returned as struct
of the defination internally as WIN32_FILE_DATA , 

we will use only its some data members like last file access time and print them to shell
for our use case of watching changes to file and sending notification to client on changes 
this much will be enough now , else there is lot more in the windows.h APIs

*/
int main(int argc,char**argv) {
//	DECLARING ALL VARIABLES AT TOP IS GOOD CODING PRACTICE IN C , see spaces given below ,its good to do like this
	char *          basePath=NULL;
	char *          token=NULL;
	WIN32_FIND_DATA FileData;           //stores attributes of the file found
	HANDLE          hSearch;            //this variable tells if the file to be watched is found or not
	DWORD           dwAttrs;
	TCHAR           szNewPath[MAX_PATH];
	short           TimeIntervalToWatchChanges=10;
	FILETIME        OldLastAccessTime;
	FILETIME        OldLastWriteTime;
	if(argc<2) {
		//below line written is to tell reader that this program need to be executed on shell or command line
		//by specifying program name and path of file to be watched
		fprintf(stderr,"USAGE : rolexhound Path\n");
		exit(EXT_WITH_TOO_FEW_ARGS);
	}
	basePath=(char*)malloc(sizeof(char)*(strlen(argv[1])+1));
	strcpy(basePath,argv[1]);
//	we dont wish to modify the path of the file
//  in the arguements itself so we are copying it into basePath
	token=strtok(basePath,"/");
	while(token!=NULL) {
		basePath=token;
		token=strtok(NULL,"/");
	}
//	printf("%s\n",basePath);
//	what above lines do is extract basePath from the absolute or relative path of file
//  executing this in cmd gives
//  rolexhound app/strcpy/ter/rolexhound.c
//  rolexhound.c

	while(true) {
		//This is main loop of server
		hSearch = FindFirstFile(basePath, &FileData);
		if (hSearch == INVALID_HANDLE_VALUE) {
			printf("No such file found.\n");
			return;
		}else{
			//Only some attributes that are useful to monitor changes inside file , like 
			//last write time , last access time will be used here for our use case
			printf("\nWatching File %s for any changes...\n",basePath);
			if(OldLastAccessTime.dwLowDateTime!=FileData.ftLastAccessTime.dwLowDateTime 
			   || OldLastAccessTime.dwHighDateTime!=FileData.ftLastAccessTime.dwHighDateTime ){
				printf("\n%s was last accessed at %lu",basePath,FileData.ftLastAccessTime);
				OldLastAccessTime.dwLowDateTime=FileData.ftLastAccessTime.dwLowDateTime;
				OldLastAccessTime.dwHighDateTime=FileData.ftLastAccessTime.dwHighDateTime;
			}if(OldLastWriteTime.dwLowDateTime!=FileData.ftLastWriteTime.dwLowDateTime 
			   || OldLastWriteTime.dwHighDateTime!=FileData.ftLastWriteTime.dwHighDateTime ){
				printf("\n%s was last written at %lu",basePath,FileData.ftLastWriteTime);
				OldLastWriteTime.dwLowDateTime=FileData.ftLastWriteTime.dwLowDateTime;
				OldLastWriteTime.dwHighDateTime=FileData.ftLastWriteTime.dwHighDateTime;
			}
			sleep(TimeIntervalToWatchChanges);
		};
	};
}


