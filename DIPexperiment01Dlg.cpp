
// DIPexperiment01Dlg.cpp: 实现文件
//

#include "pch.h"
#include "framework.h"
#include "DIPexperiment01.h"
#include "DIPexperiment01Dlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#define RGB555_MASK_RED     0x7C00
#define RGB555_MASK_GREEN   0x03E0
#define RGB555_MASK_BLUE    0x001F

static BYTE image[4][1500][1500];
static BYTE image_sew[4][1500][1500];
static BYTE palette[256][4];
static BYTE color[1];
static short bit;
static short is555;
static int index;


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CDIPexperiment01Dlg 对话框



CDIPexperiment01Dlg::CDIPexperiment01Dlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DIPEXPERIMENT01_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CDIPexperiment01Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, ID_EDIT01, EDIT01);
	DDX_Control(pDX, IDC_EDIT2, EDIT02);
	DDX_Control(pDX, ID_EDIT2, EDIT03);
}

BEGIN_MESSAGE_MAP(CDIPexperiment01Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDOK, &CDIPexperiment01Dlg::OnBnClickedOk)
	ON_BN_CLICKED(BUTTON01, &CDIPexperiment01Dlg::OnBnClickedButton01)
	ON_BN_CLICKED(BUTTON02, &CDIPexperiment01Dlg::OnBnClickedButton02)
	ON_EN_CHANGE(ID_EDIT01, &CDIPexperiment01Dlg::OnEnChangeEdit01)
	ON_EN_CHANGE(IDC_EDIT2, &CDIPexperiment01Dlg::OnEnChangeEdit2)
END_MESSAGE_MAP()


// CDIPexperiment01Dlg 消息处理程序

BOOL CDIPexperiment01Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CDIPexperiment01Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CDIPexperiment01Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CDIPexperiment01Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


// 输入文件名获得RGB
BOOL ReadBmpImage(CString fileName, BYTE image[4][1500][1500], long &height, long &width)
{
	char bmp[2];
	CFile file;
	int c;

	file.Open(fileName, CFile::modeRead);
	file.Read(bmp, 2);

	if (bmp[0] != 'B' || bmp[1] != 'M')
	{
		return 0;
	}

	file.Seek(16, CFile::current);  // 指针指向宽度信息
	file.Read(&width, 4);
	file.Read(&height, 4);
	file.Seek(2, CFile::current);
	file.Read(&bit, 2);  // 指针指向像素的颜色信息
	file.Read(&is555, 4);  // 查找16位储存类型
	file.Seek(20, CFile::current);

	// 仅有8、4、1位图像有调色板

	if (bit == 32)
	{
		for (int i = height - 1; i >= 0; i--)
		{
			for (int j = 0; j <= width - 1; j++)
			{
				for (int k = 3; k >= 0; k--)
				{
					file.Read(&image[k][i][j], 1);
				}
			}
		}
		file.Close();
		return(1);
	}

	if (bit == 24)
	{
		c = width * 3 % 4;  // 多余字节
		for (int i = height - 1; i >= 0; i--)
		{
			for (int j = 0; j <= width - 1; j++)
			{
				for (int k = 2; k >= 0; k--)
				{
					file.Read(&image[k][i][j], 1);
				}
			}
			if (c != 0)
			{
				file.Seek(4 - c, CFile::current);
			}
		}
		file.Close();
		return(1);
	}

	if (bit == 16 && is555 == BI_BITFIELDS)  // 565格式
	{
		file.Seek(12, CFile::current);
		c = width * 2 % 4;  // 多余字节
		for (int i = height - 1; i >= 0; i--)
		{
			for (int j = 0; j <= width - 1; j++)
			{
				for (int k = 0; k <= 2; k++)
				{
					if (k == 0)
					{
						file.Seek(1, CFile::current);
						file.Read(&image[k][i][j], 1);
						image[k][i][j] = (image[k][i][j] >> 3) << 3;				
					}
					else if (k == 1)
					{
						file.Seek(-1, CFile::current);
						file.Read(&image[k][i][j], 1);
						image[k][i][j] = image[k][i][j] << 5;
						file.Seek(-2, CFile::current);
						file.Read(&image_sew[k][i][j], 1);
						image_sew[k][i][j] = (image_sew[k][i][j] >> 5) << 2;
						image[k][i][j] = image[k][i][j] | image_sew[k][i][j];
					}
					else
					{
						file.Seek(-1, CFile::current);
						file.Read(&image[k][i][j], 1);
						image[k][i][j] = image[k][i][j] << 3;
					}	
				}
				file.Seek(1, CFile::current);
			}
			if (c != 0)
			{
				file.Seek(4 - c, CFile::current);
			}
		}
		file.Close();
		return(1);
	}

	if (bit == 16 && is555 == BI_RGB)  // 555格式
	{
		c = width * 2 % 4;  // 多余字节
		for (int i = height - 1; i >= 0; i--)
		{
			for (int j = 0; j <= width - 1; j++)
			{
				for (int k = 0; k <= 2 ; k++)
				{
					if (k == 0)
					{
						file.Seek(1, CFile::current);
						file.Read(&image[k][i][j], 1);
						image[k][i][j] = (image[k][i][j] >> 2) << 3;
					}
					else if (k == 1)
					{
						file.Seek(-1, CFile::current);
						file.Read(&image[k][i][j], 1);
						image[k][i][j] = image[k][i][j] << 6;
						file.Seek(-2, CFile::current);
						file.Read(&image_sew[k][i][j], 1);
						image_sew[k][i][j] = (image_sew[k][i][j] >> 5) << 3;
						image[k][i][j] = image[k][i][j] | image_sew[k][i][j];
					}
					else
					{
						file.Seek(-1, CFile::current);
						file.Read(&image[k][i][j], 1);
						image[k][i][j] = image[k][i][j] << 3;
					}
				}
				file.Seek(1, CFile::current);
			}
			if (c != 0)
			{
				file.Seek(4 - c, CFile::current);
			}
		}
		file.Close();
		return(1);
	}


	if (bit == 8)
	{
		c = width % 4;  // 多余字节
		// 给调色盘赋值
		for (int x = 0; x < 256; x++)
		{
			for (int y = 0; y < 4; y++)
			{
				file.Read(&palette[x][y], 1);
			}
		}
		for (int i = height - 1; i >= 0; i--)
		{
			for (int j = 0; j <= width - 1; j++)
			{
				file.Read(&index, 1);
				for (int k = 2; k >= 0; k--)
				{				
					image[k][i][j] = palette[index][2 - k];
				}
			}
			if (c != 0)
			{
				file.Seek(4 - c, CFile::current);
			}
		}
		file.Close();
		return(1);
	}

	if (bit == 4)
	{
		c = (width / 2 + 1) % 4;  // 多余字节
		// 给调色盘赋值
		for (int x = 0; x < 16; x++)
		{
			for (int y = 0; y < 4; y++)
			{
				file.Read(&palette[x][y], 1);
			}
		}
		for (int i = height - 1; i >= 0; i--)
		{
			for (int j = 0; j <= width - 1; j++)
			{
				file.Read(&index, 1);
				for (int k = 2; k >= 0; k--)
				{
					image[k][i][j] = palette[index][2 - k];
				}
			}
		}
		file.Close();
		return(1);
	}

	if (bit == 2)
	{
		c = (width / 4 + 1) % 4;  // 多余字节
		// 给调色盘赋值
		for (int x = 0; x < 4; x++)
		{
			for (int y = 0; y < 4; y++)
			{
				file.Read(&palette[x][y], 1);
			}
		}
		for (int i = height - 1; i >= 0; i--)
		{
			for (int j = 0; j <= width - 1; j++)
			{
				file.Read(&index, 1);
				for (int k = 2; k >= 0; k--)
				{
					image[k][i][j] = palette[index][2 - k];
				}
			}
			if (c != 0)
			{
				file.Seek(4 - c, CFile::current);
			}
		}
		file.Close();
		return(1);
	}

	if (bit == 1)
	{
		c = (width / 8 + 1) % 4;  // 多余字节
		// 给调色盘赋值
		for (int x = 0; x < 2; x++)
		{
			for (int y = 0; y < 4; y++)
			{
				file.Read(&palette[x][y], 1);
			}
		}
		for (int i = height - 1; i >= 0; i--)
		{
			for (int j = 0; j <= width - 1; j++)
			{
				file.Read(&index, 1);
				for (int k = 2; k >= 0; k--)
				{
					image[k][i][j] = palette[index][2 - k];
				}
			}
			if (c != 0)
			{
				file.Seek(4 - c, CFile::current);
			}
		}
		file.Close();
		return(1);
	}

}

// 显示图像
void DispColorImage(CDC* p, BYTE image[4][1500][1500], long height, long width, int dx, int dy)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			p->SetPixel(j + dx, i + dy, RGB(image[0][i][j], image[1][i][j], image[2][i][j]));
		}
	}	
}

void CDIPexperiment01Dlg::OnBnClickedOk()
{
	// TODO: 在此添加控件通知处理程序代码
	CDialogEx::OnOK();
}


void CDIPexperiment01Dlg::OnBnClickedButton01()
{
	// TODO: 在此添加控件通知处理程序代码
	long height, width;
	CFileDialog dlg(true);
	if (dlg.DoModal() == IDOK)
	{
		CDC* p = GetDlgItem(ID_IMAG)->GetDC();
		ReadBmpImage(dlg.GetFileName(), image, height, width);
		DispColorImage(p, image, height, width, 0, 0);  // dx dy表示相比左上角的偏移量
	}
	else
		return;
}


void CDIPexperiment01Dlg::OnBnClickedButton02()
{
	// TODO: 在此添加控件通知处理程序代码
	CString str1, str2, strR, strG, strB;
	long height, width;
	CFileDialog dlg(true);

	int x, y;
	EDIT02.GetWindowText(str1);
	x = _ttol(str1) - 1;
	EDIT01.GetWindowText(str2);
	y = _ttol(str2) - 1;

	strR.Format(_T("%d"), image[0][x][y]);
	strG.Format(_T("%d"), image[1][x][y]);
	strB.Format(_T("%d"), image[2][x][y]);
	
	EDIT03.SetWindowText(strR + " " + strG + " " + strB);
}


void CDIPexperiment01Dlg::OnEnChangeEdit1()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}


void CDIPexperiment01Dlg::OnEnChangeEdit01()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}


void CDIPexperiment01Dlg::OnEnChangeEdit2()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
}
